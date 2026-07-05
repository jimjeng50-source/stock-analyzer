"""
utils/gdrive.py
Google Drive 上傳（服務帳號）

前置設定（一次性）：
1. Google Cloud Console 建立專案 → 啟用 Google Drive API
2. 建立服務帳號 → 金鑰（JSON）
3. 在 Google Drive 建一個資料夾，「共用」給服務帳號的 email（編輯者）
4. 設定兩個 secret：
   - GDRIVE_SERVICE_ACCOUNT_JSON：整份 JSON 金鑰內容
   - GDRIVE_FOLDER_ID：資料夾網址最後一段 ID

未設定憑證時 upload_file() 回傳 False 並記 log，不拋例外。
"""

import json
import logging
import mimetypes
import os

import requests

from config import get_runtime_config

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
_LIST_URL = "https://www.googleapis.com/drive/v3/files"
_SCOPE = "https://www.googleapis.com/auth/drive.file"


def is_configured() -> bool:
    """是否已設定 Drive 憑證。"""
    return bool(get_runtime_config("GDRIVE_SERVICE_ACCOUNT_JSON")
                and get_runtime_config("GDRIVE_FOLDER_ID"))


def _get_access_token() -> str:
    """用服務帳號 JWT 換取 access token。"""
    import time
    import base64
    import hashlib

    creds = json.loads(get_runtime_config("GDRIVE_SERVICE_ACCOUNT_JSON"))

    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError as e:
        raise RuntimeError("需要 cryptography 套件（pip install cryptography）") from e

    def _b64(data: bytes) -> bytes:
        return base64.urlsafe_b64encode(data).rstrip(b"=")

    now = int(time.time())
    header = _b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    claims = _b64(json.dumps({
        "iss": creds["client_email"],
        "scope": _SCOPE,
        "aud": _TOKEN_URL,
        "iat": now,
        "exp": now + 3600,
    }).encode())
    signing_input = header + b"." + claims

    key = serialization.load_pem_private_key(creds["private_key"].encode(), password=None)
    signature = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    jwt = (signing_input + b"." + _b64(signature)).decode()

    resp = requests.post(_TOKEN_URL, data={
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": jwt,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def upload_file(local_path: str, drive_name: str = None) -> bool:
    """
    上傳（或更新）檔案到設定的 Drive 資料夾。
    同名檔案存在時更新內容（保持同一個檔案連結）。
    """
    if not is_configured():
        logger.info("Google Drive 未設定（GDRIVE_SERVICE_ACCOUNT_JSON / GDRIVE_FOLDER_ID），跳過上傳")
        return False
    if not os.path.exists(local_path):
        logger.warning("Drive 上傳失敗：檔案不存在 %s", local_path)
        return False

    drive_name = drive_name or os.path.basename(local_path)
    folder_id = get_runtime_config("GDRIVE_FOLDER_ID")
    mime = mimetypes.guess_type(local_path)[0] or "application/octet-stream"

    try:
        token = _get_access_token()
        headers = {"Authorization": f"Bearer {token}"}

        # 找同名既有檔案
        q = f"name = '{drive_name}' and '{folder_id}' in parents and trashed = false"
        resp = requests.get(_LIST_URL, params={"q": q, "fields": "files(id)"},
                            headers=headers, timeout=30)
        resp.raise_for_status()
        existing = resp.json().get("files", [])

        with open(local_path, "rb") as f:
            content = f.read()

        if existing:
            # 更新既有檔案內容
            file_id = existing[0]["id"]
            resp = requests.patch(
                f"https://www.googleapis.com/upload/drive/v3/files/{file_id}?uploadType=media",
                headers={**headers, "Content-Type": mime},
                data=content, timeout=60,
            )
        else:
            # 建立新檔（multipart：metadata + content）
            metadata = json.dumps({"name": drive_name, "parents": [folder_id]})
            files = {
                "metadata": ("metadata", metadata, "application/json"),
                "file": (drive_name, content, mime),
            }
            resp = requests.post(_UPLOAD_URL, headers=headers, files=files, timeout=60)

        resp.raise_for_status()
        logger.info("Google Drive 上傳成功：%s", drive_name)
        return True

    except Exception as e:
        logger.error("Google Drive 上傳失敗：%s", e)
        return False
