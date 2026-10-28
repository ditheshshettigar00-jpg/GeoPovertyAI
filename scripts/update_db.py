import sqlite3

def update_db():
    conn = sqlite3.connect('database/geopoverty.db')
    conn.execute('DROP TABLE IF EXISTS users')
    conn.commit()
    with open('database/schema.sql', 'r') as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    print('Database schema updated')

if __name__ == '__main__':
    update_db()
