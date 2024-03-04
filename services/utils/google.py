try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

import os
import csv
import json
import io

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError


if os.getenv('IS_OFFLINE') or not os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
   LAMBDA_DATA_DIR = '.'
else:
   LAMBDA_DATA_DIR = '/tmp'


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


def list_files_in_drive_folder(folder_id=None, creds=None, include=[]):
    SUFFIX_TO_MIME = {
        'csv': 'text/csv',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'txt': 'text/plain',
        'md': 'text/markdown',
        'pdf': 'application/pdf',
        'json': 'application/vnd.google-apps.script+json',
        'folders': 'application/vnd.google-apps.folder'
    }

    service = build("drive", "v3", credentials=creds)

    parent_q = f"'{folder_id}' in parents"

    if len(include) > 0:
        include_mimes = ' or '.join([f"mimeType = '{SUFFIX_TO_MIME[mime_type]}'" for mime_type in include])
        include_mimes = f"({include_mimes})"
        q = f"({include_mimes} and {parent_q})"
    else:
        q = parent_q
    
    files = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=q,
                spaces="drive",
                fields="nextPageToken, files(id, name)",
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if page_token is None:
            break

    return files


def download_from_drive(file_id: str, local_file_name: str, creds):
  """Downloads a file
  Args:
      real_file_id: ID of the file to download
  Returns : IO object with location.
  """

  try:
      # create drive api client
      service = build("drive", "v3", credentials=creds)

      # pylint: disable=maybe-no-member
      request = service.files().get_media(fileId=file_id)
      file = io.BytesIO()
      downloader = MediaIoBaseDownload(file, request)
      done = False
      while done is False:
          status, done = downloader.next_chunk()

      with open(local_file_name, "wb") as f:
          f.write(file.getvalue())

  except HttpError as error:
      print(f"An error occurred: {error}")
      file = None

  return


def create_drive_folder(name, parents=None, creds=None):
    service = build("drive", "v3", credentials=creds)
    
    folder_metadata = {
        "name": name,
        "parents": [parents],
        "mimeType": "application/vnd.google-apps.folder",
    }

    folder = service.files().create(body=folder_metadata, fields="id").execute()
    folder_id = folder.get("id")
    print(f'Folder ID: "{folder_id}".')
    
    return folder_id


def search_drive_folders(parent_id=None, creds=None):
    service = build("drive", "v3", credentials=creds)

    if parent_id and parent_id != 'root':
        parent_q = f"'{parent_id}' in parents"
        q = f"(trashed = false and mimeType = 'application/vnd.google-apps.folder' and {parent_q}) "
    else:
        q = ("(trashed = false and mimeType = 'application/vnd.google-apps.folder' and 'root' in parents) " +
            "or (sharedWithMe and trashed = false and mimeType = 'application/vnd.google-apps.folder')")
    
    files = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=q,
                spaces="drive",
                fields="nextPageToken, files(id, name)",
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if page_token is None:
            break

    return files


def create_google_doc(title, content='', folder_id='root', creds=None):
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

    # Move the file to the new folder
    service = build("drive", "v3", credentials=creds)
    file = (
        service.files()
        .update(
            fileId=doc_id,
            addParents=folder_id,
            removeParents='root',
            fields="id, parents",
        )
        .execute()
    )
    print('Created Google Doc with title: {}'.format(doc.get('title')))
    
    return doc_id


def get_csv_lines(content=None, path=None, delimiter=',', return_as_json=False):
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

    if return_as_json:
        data = [json.dumps({h: v for h, v in zip(data[0], row)}) for row in data[1:]]

    return data


def create_google_sheet(title, content=None, path=None, delimiter=',', folder_id='root', creds=None):
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

    # Move the file to the new folder
    service = build("drive", "v3", credentials=creds)
    file = (
        service.files()
        .update(
            fileId=spreadsheet_id,
            addParents=folder_id,
            removeParents='root',
            fields="id, parents",
        )
        .execute()
    )
    print('Spreadsheet ID: {}'.format(spreadsheet_id))
    
    return spreadsheet_id


def upload_to_google_drive(title, file_type, content=None, path=None, folder_id='root', creds=None):
    service = build("drive", "v3", credentials=creds)

    file_metadata = {
        'name': title,
        'parents': [folder_id]
    }

    if content:
        file_name = f"{title}.{file_type}"
        path = os.path.join(LAMBDA_DATA_DIR, file_name)
        with open(path, 'w') as file:
            file.write(content)

    if file_type == 'md':
        mimetype = 'text/markdown'
    elif file_type == 'txt':
        mimetype = 'text/plain'
    elif file_type == 'csv':
        mimetype = 'text/csv'
    else:
        mimetype = f'application/{file_type}'

    media = MediaFileUpload(path, mimetype=mimetype)

    # Upload the file
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get("id")
    print(f'File ID: {file_id}')

    if content:
        os.remove(path)
    
    return file_id


def convert_text(text, from_format='md', to_format='rtf'):
    import pypandoc

    output = pypandoc.convert_text(text, to_format, format=from_format, extra_args=['-s'])
    #_ = pypandoc.convert_file('TestAPI.md', 'rtf', outputfile="TestAPI.rtf", extra_args=['-s'])

    return output