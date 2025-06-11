#!/usr/bin/env python3
"""
convert_tsv_to_sheets.py

Convert TSV files in a local directory (or specified files) into individual Google Sheets,
replicating local subfolder structure in the target Drive folder, with header styling,
alignment, column widths, retry logic for transient errors, and underscore-based header wrapping.
Deletes uploaded files and empties folders afterward if --delete-original is passed.
"""
import os
import argparse
import csv
import logging
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

SCOPES = [
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/spreadsheets'
]

logger = logging.getLogger(__name__)

def execute_request_with_retries(func, max_retries=5, initial_delay=1, backoff_factor=2):
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except HttpError as e:
            status = getattr(e.resp, 'status', None)
            if status in (429, 500, 502, 503, 504):
                logger.warning(f"API error {status}, attempt {attempt}/{max_retries}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"Non-retriable HttpError {status}: {e}")
                raise
    raise RuntimeError(f"Failed API request after {max_retries} attempts.")

def parse_tsv_file(path):
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter='\t', quotechar='"')
        return [row for row in reader]

def get_column_widths_for_file(file_name, num_columns):
    name = file_name.lower()
    widths = None
    if 'refine' in name and ('zeroshot_cot' in name or 'zscot' in name):
        widths = [45,150,150,60,200,400,400] + [75]*14
    elif 'refine' in name and ('zeroshot' in name or 'zs' in name):
        widths = [45,150,150,60,200,400,400] + [75]*14
    elif 'refine' in name and ('fewshot_cot' in name or 'fscot' in name):
        widths = [45,150,150,60,200,400,400] + [75]*14
    elif 'stats' in name and 'vs' in name:
        widths = [150,150,150,150]
    elif 'vs' in name:
        widths = [45,140,125,60,300,250,300] + [75]*3
    elif 'zeroshot_with_knowledge' in name or 'zsk' in name:
        widths = [45,160,160,60,250,400] + [75]*5
    elif 'zeroshot_cot_with_knowledge' in name or 'zscotk' in name or 'zscot_with_knowledge' in name:
        widths = [45,160,160,60,350,300] + [75]*5
    elif 'zeroshot_cot' in name or 'zscot' in name:
        widths = [45,300,200,100,400] + [75]*5
    elif 'zeroshot' in name or 'zs' in name:
        widths = [45,300,200,100,400] + [75]*5
    elif 'fewshot_with_knowledge' in name or 'fsk' in name:
        widths = [45,300,200,100,100,325] + [75]*5
    elif 'fewshot_cot' in name or 'fscot' in name:
        widths = [45,300,200,100,400] + [75]*5
    elif 'fewshot' in name or 'fs' in name:
        widths = [45,400,300,100,100] + [75]*5
    elif 'accuracy' in name:
        widths = [200,100,100,100,100,100,100]
    else:
        widths = [100] * num_columns
    if len(widths) < num_columns:
        widths.extend([100] * (num_columns - len(widths)))
    return widths[:num_columns]

def get_services(credentials_json):
    creds = service_account.Credentials.from_service_account_file(credentials_json, scopes=SCOPES)
    sheets_svc = build('sheets', 'v4', credentials=creds)
    drive_svc = build('drive', 'v3', credentials=creds)
    return sheets_svc, drive_svc

def get_or_create_drive_folder(drive_svc, parent_id, folder_name):
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and '{parent_id}' in parents and trashed=false"
    resp = execute_request_with_retries(lambda: drive_svc.files().list(q=query, fields='files(id,name)').execute())
    files = resp.get('files', [])
    if files:
        return files[0]['id']
    metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_id]}
    folder = execute_request_with_retries(lambda: drive_svc.files().create(body=metadata, fields='id').execute())
    logger.info(f"Created Drive folder: {folder_name} (ID: {folder['id']})")
    return folder['id']

def upload_to_sheets(sheets_svc, drive_svc, rows, title, root_input, root_drive, file_path):
    # 1) Create the sheet
    spreadsheet = execute_request_with_retries(lambda: sheets_svc.spreadsheets().create(
        body={'properties': {'title': title}},
        fields='spreadsheetId,sheets(properties(sheetId))'
    ).execute())
    ss_id   = spreadsheet['spreadsheetId']
    sheet_id = spreadsheet['sheets'][0]['properties']['sheetId']

    # 2) Push raw values (as you had)
    execute_request_with_retries(lambda: sheets_svc.spreadsheets().values().update(
        spreadsheetId=ss_id, range='A1',
        valueInputOption='RAW', body={'values': rows}
    ).execute())

    # 3) Compute widths
    num_cols = len(rows[0])
    widths   = get_column_widths_for_file(title, num_cols)

    # 4) First—adjust the header text to swap '_' → ' ' or '\n' based on width
    #    (mimics your adjustAllHeaders)
    #    Fetch current header row:
    header_resp = execute_request_with_retries(lambda:
        sheets_svc.spreadsheets().values().get(
            spreadsheetId=ss_id, range='1:1'
        ).execute()
    )
    header = header_resp.get('values', [[]])[0]
    avg_char = 7
    padding  = 20
    for i, cell in enumerate(header):
        needed = len(cell) * avg_char + padding
        if needed <= widths[i]:
            header[i] = cell.replace('_', ' ')
        else:
            header[i] = cell.replace('_', '\n')
    #    Write the *adjusted* header back with USER_ENTERED (so newlines & wrap take effect)
    end_col_letter = chr(ord('A') + len(header) - 1)
    execute_request_with_retries(lambda: sheets_svc.spreadsheets().values().update(
        spreadsheetId=ss_id,
        range=f"A1:{end_col_letter}1",
        valueInputOption='USER_ENTERED',
        body={'values': [header]}
    ).execute())

    # 5) Build a single batchUpdate to handle:
    #    • text-as-text (@)
    #    • wrapping & top-left align
    #    • bold first row
    #    • freeze header
    #    • column widths
    requests = []

    # 5a) force text format on entire sheet
    requests.append({
        'repeatCell': {
            'range': {'sheetId': sheet_id},
            'cell': {
                'userEnteredFormat': {
                    'numberFormat': {'type': 'TEXT'}
                }
            },
            'fields': 'userEnteredFormat.numberFormat'
        }
    })

    # 5b) wrap + top-left align everywhere
    requests.append({
        'repeatCell': {
            'range': {'sheetId': sheet_id},
            'cell': {
                'userEnteredFormat': {
                    'wrapStrategy':       'WRAP',
                    'horizontalAlignment': 'LEFT',
                    'verticalAlignment':   'TOP'
                }
            },
            'fields': 'userEnteredFormat(wrapStrategy,horizontalAlignment,verticalAlignment)'
        }
    })

    # 5c) bold only the header row
    requests.append({
        'repeatCell': {
            'range': {
                'sheetId':       sheet_id,
                'startRowIndex': 0,
                'endRowIndex':   1
            },
            'cell': {
                'userEnteredFormat': {
                    'textFormat': {'bold': True}
                }
            },
            'fields': 'userEnteredFormat.textFormat.bold'
        }
    })

    # 5d) freeze the first row
    requests.append({
        'updateSheetProperties': {
            'properties': {
                'sheetId': sheet_id,
                'gridProperties': {'frozenRowCount': 1}
            },
            'fields': 'gridProperties.frozenRowCount'
        }
    })

    # 5e) set each column width
    for idx, px in enumerate(widths):
        requests.append({
            'updateDimensionProperties': {
                'range': {
                    'sheetId':  sheet_id,
                    'dimension': 'COLUMNS',
                    'startIndex': idx,
                    'endIndex':   idx + 1
                },
                'properties': {'pixelSize': px},
                'fields': 'pixelSize'
            }
        })

    # 6) Execute all formatting in one shot
    execute_request_with_retries(lambda: sheets_svc.spreadsheets()
        .batchUpdate(spreadsheetId=ss_id, body={'requests': requests})
        .execute())

    # 7) Finally, move the sheet into your Drive structure (as you already do)
    rel_path = os.path.relpath(os.path.dirname(file_path), root_input)
    parent = root_drive
    if rel_path not in ('.', ''):
        for part in rel_path.split(os.sep):
            logger.warning(f"Creating/checking folder '{part}' under parent '{parent}'")
            parent = get_or_create_drive_folder(drive_svc, parent, part)
    execute_request_with_retries(lambda: drive_svc.files().update(
        fileId=ss_id,
        addParents=parent,
        removeParents='root',
        fields='id,parents'
    ).execute())
    return ss_id


def process_file(path, sheets_svc, drive_svc, root_input, root_drive, delete_original=False):
    fname = os.path.basename(path)
    base, _ = os.path.splitext(fname)
    logger.info(f"Processing TSV: {fname}")

    rows = parse_tsv_file(path)
    if not rows:
        logger.warning(f"{fname} is empty, skipping.")
        return

    upload_to_sheets(sheets_svc, drive_svc, rows, title=base,
                     root_input=root_input, root_drive=root_drive, file_path=path)

    if delete_original:
        os.remove(path)
        logger.info(f"Deleted original TSV: {fname}")
        parent = os.path.dirname(path)
        while parent and os.path.isdir(parent) and not os.listdir(parent):
            os.rmdir(parent)
            logger.info(f"Deleted empty folder: {parent}")
            parent = os.path.dirname(parent)

def main():
    parser = argparse.ArgumentParser(description="Convert TSV files to Google Sheets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-dir', help='Directory to search for .tsv')
    group.add_argument('--files', nargs='+', help='Specific TSV files')
    parser.add_argument('--drive-folder-id', required=True, help='Root Drive folder ID')
    parser.add_argument('--credentials', required=True, help='Service account JSON')
    parser.add_argument('--delete-original', action='store_true')
    parser.add_argument('--debug', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    logging_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level=logging_level, format='%(asctime)s %(levelname)s: %(message)s')

    sheets_svc, drive_svc = get_services(args.credentials)
    root_input = args.input_dir or os.path.commonpath(args.files)
    root_drive = args.drive_folder_id
    tsvs = args.files if args.files else [os.path.join(root, fn)
                 for root, _, fns in os.walk(root_input) for fn in fns if fn.lower().endswith('.tsv')]

    for t in tqdm(tsvs, desc="Uploading files", unit="file"):
        process_file(t, sheets_svc, drive_svc,
                     root_input=root_input, root_drive=root_drive,
                     delete_original=args.delete_original)
        
    # ─── remove any now-empty subfolders under the input root ───
    for dirpath, dirnames, filenames in os.walk(root_input, topdown=False):
        # skip the root_input itself
        if dirpath == root_input:
            logger.info(f"This is root.")
            continue
        if not os.listdir(dirpath):
            try:
                os.rmdir(dirpath)
                logger.info(f"Removed empty folder: {dirpath}")
            except OSError as e:
                logger.warning(f"Could not remove folder {dirpath}: {e}")

if __name__ == '__main__':
    main()
