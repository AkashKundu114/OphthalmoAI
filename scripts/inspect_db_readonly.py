import sqlite3
import os
import re

def inspect_file(db_file):
    db_path = os.path.abspath(db_file)
    if not os.path.exists(db_path):
        print(f"Database file {db_file} not found.")
        return
    print(f"\n=======================================================")
    print(f"Inspecting: {db_file} (Size: {os.path.getsize(db_path):,} bytes)")
    print(f"=======================================================")
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall() if not r[0].startswith("sqlite_")]
    print(f"Total Non-System Tables: {len(tables)}\n")
    user_accounts_present = False
    password_hashes_present = False
    emails_present = False
    tokens_present = False
    scan_records_present = False
    audit_logs_present = False
    image_paths_present = False
    personal_data_detected = False

    for t in sorted(tables):
        cursor.execute(f"PRAGMA table_info({t});")
        cols = [col[1] for col in cursor.fetchall()]
        cursor.execute(f"SELECT COUNT(*) FROM {t};")
        cnt = cursor.fetchone()[0]
        print(f"Table: {t:25} | Rows: {cnt:5} | Columns: {', '.join(cols)}")
        if "user" in t.lower() and cnt > 0: user_accounts_present = True
        if any(c in cols for c in ["hashed_password", "password_hash", "password"]) and cnt > 0: password_hashes_present = True
        if any(c in cols for c in ["email", "user_email"]) and cnt > 0: emails_present = True
        if any(c in cols for c in ["token", "refresh_token", "jwt"]) and cnt > 0: tokens_present = True
        if any(k in t.lower() for k in ["scan", "prediction", "case"]) and cnt > 0: scan_records_present = True
        if "audit" in t.lower() and cnt > 0: audit_logs_present = True
        if any(c in cols for c in ["image_path", "file_path", "storage_path"]) and cnt > 0: image_paths_present = True

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
    if cursor.fetchone():
        cursor.execute("SELECT email, role FROM users LIMIT 10;")
        users = cursor.fetchall()
        print("\n--- Inspecting users table (emails sanitized) ---")
        for u in users:
            email = u[0]
            role = u[1]
            is_test = bool(re.search(r"test|mock|example|demo|admin@hospital\.org|clinician@", email, re.I))
            print(f"User email pattern: {email.split('@')[0][:3]}***@{email.split('@')[-1]} | Role: {role} | Looks synthetic/fixture: {is_test}")
            if not is_test:
                personal_data_detected = True

    conn.close()
    print("\n--- Summary Checklist ---")
    print(f"User accounts present: {user_accounts_present}")
    print(f"Password hashes present: {password_hashes_present}")
    print(f"Emails present: {emails_present}")
    print(f"Tokens present: {tokens_present}")
    print(f"Scan records present: {scan_records_present}")
    print(f"Audit logs present: {audit_logs_present}")
    print(f"Image paths present: {image_paths_present}")
    print(f"Contains personal or sensitive data: {'YES' if personal_data_detected else 'NO (only synthetic fixtures / hashes from tests)'}")

inspect_file("ophthalmoai.db")
inspect_file("test_ophthalmoai.db")
exit(0)

uri = f"file:{db_path}?mode=ro"
conn = sqlite3.connect(uri, uri=True)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [r[0] for r in cursor.fetchall() if not r[0].startswith("sqlite_")]

print(f"Database Path: {db_path}")
print(f"Total Non-System Tables: {len(tables)}\n")

user_accounts_present = False
password_hashes_present = False
emails_present = False
tokens_present = False
scan_records_present = False
audit_logs_present = False
image_paths_present = False
personal_data_detected = False

for t in sorted(tables):
    cursor.execute(f"PRAGMA table_info({t});")
    cols = [col[1] for col in cursor.fetchall()]
    cursor.execute(f"SELECT COUNT(*) FROM {t};")
    cnt = cursor.fetchone()[0]
    print(f"Table: {t:25} | Rows: {cnt:5} | Columns: {', '.join(cols)}")

    if "user" in t.lower() and cnt > 0:
        user_accounts_present = True
    if any(c in cols for c in ["hashed_password", "password_hash", "password"]) and cnt > 0:
        password_hashes_present = True
    if any(c in cols for c in ["email", "user_email"]) and cnt > 0:
        emails_present = True
    if any(c in cols for c in ["token", "refresh_token", "jwt"]) and cnt > 0:
        tokens_present = True
    if any(k in t.lower() for k in ["scan", "prediction", "case"]) and cnt > 0:
        scan_records_present = True
    if "audit" in t.lower() and cnt > 0:
        audit_logs_present = True
    if any(c in cols for c in ["image_path", "file_path", "storage_path"]) and cnt > 0:
        image_paths_present = True

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
if cursor.fetchone():
    cursor.execute("SELECT email, role FROM users LIMIT 10;")
    users = cursor.fetchall()
    print("\n--- Inspecting users table (emails sanitized) ---")
    for u in users:
        email = u[0]
        role = u[1]
        is_test = bool(re.search(r"test|mock|example|demo|admin@hospital\.org|clinician@", email, re.I))
        print(f"User email pattern: {email.split('@')[0][:3]}***@{email.split('@')[-1]} | Role: {role} | Looks synthetic/fixture: {is_test}")
        if not is_test:
            personal_data_detected = True

conn.close()

print("\n--- Summary Checklist ---")
print(f"User accounts present: {user_accounts_present}")
print(f"Password hashes present: {password_hashes_present}")
print(f"Emails present: {emails_present}")
print(f"Tokens present: {tokens_present}")
print(f"Scan records present: {scan_records_present}")
print(f"Audit logs present: {audit_logs_present}")
print(f"Image paths present: {image_paths_present}")
print(f"Contains personal or sensitive data: {'YES' if personal_data_detected else 'NO (only synthetic fixtures / hashes from tests)'}")
