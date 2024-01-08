try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

import os
import pypandoc
import csv

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def get_creds(goog_token, goog_refresh_token):
    creds = Credentials(
        token=goog_token,
        refresh_token=goog_refresh_token,
        token_uri="https://www.googleapis.com/oauth2/v3/token", 
        client_id=os.environ['GOOG_CLIENT_ID'],
        client_secret=os.environ['GOOG_CLIENT_SECRET']
    )

    if creds.expired and creds.refresh_token:
        print('Refreshing Token')
        creds.refresh(Request())

    return creds


def create_google_doc(title, content='', creds=None):
    service = build("docs", "v1", credentials=creds)

    body = {
        'title': title
    }
    doc = service.documents()\
        .create(body=body)\
        .execute()
    
    doc_id = doc.get('documentId')
    
    requests = [
        {
            'insertText': {
                'location': {
                    'index': 1,
                },
                'text': content
            }
        }
    ]
    service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()
    print('Created Google Doc with title: {}'.format(doc.get('title')))
    
    return doc_id


def get_csv_lines(content=None, path=None, delimiter=','):
    if path:
        data = []
        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                data.append(row)

    if content:
        data = []
        reader = csv.reader(content.splitlines(), delimiter=delimiter)
        for row in reader:
            data.append(row)

    return data


def create_google_sheet(title, content=None, path=None, delimiter=',', creds=None):
    service = build('sheets', 'v4', credentials=creds)

    # Create a new spreadsheet
    spreadsheet = {
        'properties': {
            'title': title
        }
    }
    spreadsheet = service.spreadsheets()\
        .create(body=spreadsheet, fields='spreadsheetId')\
        .execute()
    
    spreadsheet_id = spreadsheet.get('spreadsheetId')

    # Read CSV file
    csv_lines = get_csv_lines(content, path, delimiter)

    # Prepare data for Google Sheets
    body = {
        'values': csv_lines
    }
    range = 'A1' # Starting cell for data upload
    value_input_option = 'RAW' # 'RAW' or 'USER_ENTERED'

    # Upload data to the spreadsheet
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=range,
        valueInputOption=value_input_option,
        body=body
    ).execute()
    print('Spreadsheet ID: {0}'.format(spreadsheet_id))
    
    return spreadsheet_id


def upload_to_google_drive(title, file_type, content=None, path=None, creds=None):
    service = build("drive", "v3", credentials=creds)

    file_metadata = {'name': title}

    if content:
        path = f"{title}.{file_type}"
        with open(path, 'w') as file:
            file.write(content)

    media = MediaFileUpload(path, mimetype=f'application/{file_type}')

    # Upload the file
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get("id")
    print(f'File ID: {file_id}')

    if content:
        os.remove(path)
    
    return file_id


def convert_text(text, from_format='md', to_format='rtf'):
    output = pypandoc.convert_text(text, to_format, format=from_format, extra_args=['-s'])
    #_ = pypandoc.convert_file('TestAPI.md', 'rtf', outputfile="TestAPI.rtf", extra_args=['-s'])

    return output