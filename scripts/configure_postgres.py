"""Interactive local setup; administrator passwords never enter files or logs."""
import getpass
import json
from pathlib import Path
import secrets
import sys
import urllib.request

ROOT = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from neurons.storage import protect, IndividualStore

RUNTIME = ROOT/'.runtime'


def status(stage, message):
    RUNTIME.mkdir(exist_ok=True)
    (RUNTIME/'database-setup-status.json').write_text(json.dumps({'stage': stage, 'message': message}), encoding='utf8')
    print(message, flush=True)


def main():
    print('Neurons PostgreSQL setup - dedicated database and login: neurons')
    source = ROOT/'.data'/'individuals'/'individuals.sqlite3'
    if source.exists():
        try:
            with urllib.request.urlopen('http://127.0.0.1:8877/api/individuals', timeout=2) as response:
                active = json.load(response).get('ready')
        except (OSError, ValueError):
            active = False
        if active:
            raise ValueError('Close Neurons before migrating existing individuals to PostgreSQL.')
    status('waiting_for_admin', 'Enter the existing local postgres administrator password. Input is hidden.')
    password = getpass.getpass('postgres password: ')
    status('connecting_admin', 'Checking administrator connection...')
    admin = psycopg.connect(host='127.0.0.1', port=5432, user='postgres', dbname='postgres',
                           password=password, connect_timeout=5, autocommit=True)
    del password
    pending = RUNTIME/'database-pending.json'
    with admin:
        role = admin.execute("SELECT rolsuper,rolcreatedb,rolcreaterole,rolcanlogin FROM pg_roles WHERE rolname='neurons'").fetchone()
        database = admin.execute("SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname='neurons'").fetchone()
        if database and database[0] != 'neurons':
            raise ValueError('The existing neurons database has another owner. It was not changed.')
        if role and (any(role[:3]) or not role[3]):
            raise ValueError('The existing neurons role is not a dedicated limited login. It was not changed.')
        if role and pending.exists():
            dsn = protect(json.loads(pending.read_text(encoding='utf8'))['protected_dsn'], decrypt=True)
        else:
            app_password = getpass.getpass('Existing neurons login password (will not be changed): ') if role else secrets.token_urlsafe(32)
            dsn = make_conninfo(host='127.0.0.1', port=5432, user='neurons', dbname='neurons', password=app_password)
            # Save before CREATE ROLE so an interruption cannot lose a new credential.
            pending.write_text(json.dumps({'protected_dsn': protect(dsn)}), encoding='utf8')
            if not role:
                admin.execute(sql.SQL('CREATE ROLE neurons LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {}').format(sql.Literal(app_password)))
            del app_password
        if not database:
            admin.execute('CREATE DATABASE neurons OWNER neurons')
    status('checking_application', 'Checking dedicated account and tables...')
    test = IndividualStore(ROOT/'.data', dsn)
    try:
        if source.exists() and not test.all_states():
            status('migrating', 'Copying existing standalone individuals; source files are preserved.')
            test.import_sqlite(source)
    finally:
        test.close()
    temporary = RUNTIME/'database-config.tmp'
    temporary.write_text(json.dumps({'backend': 'postgresql', 'database': 'neurons', 'user': 'neurons',
                                    'protected_dsn': protect(dsn)}, indent=2), encoding='utf8')
    temporary.replace(RUNTIME/'database.json')
    status('complete', 'Connected as neurons to database neurons. Connection encrypted for this Windows account. Restart Neurons to apply.')


def run():
    try:
        main()
    except KeyboardInterrupt:
        status('cancelled', 'Setup cancelled. Existing passwords were not changed.')
        sys.exit(1)
    except psycopg.Error as error:
        message = ('Password authentication failed. Enter the existing PostgreSQL administrator password.'
                   if error.sqlstate == '28P01' or 'password authentication failed' in str(error)
                   else 'PostgreSQL setup failed. Check the service, login permissions and database ownership.')
        status('failed', message + ' SQLSTATE=' + (error.sqlstate or 'connection'))
        sys.exit(1)
    except ValueError as error:
        status('failed', str(error))
        sys.exit(1)
    except Exception as error:
        status('failed', 'Local setup failed: ' + type(error).__name__)
        sys.exit(1)


if __name__ == '__main__':
    run()
