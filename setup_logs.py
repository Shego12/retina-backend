import sqlite3

# Connect to your existing database
conn = sqlite3.connect("retina.db")
cursor = conn.cursor()

# Create the attendance_logs table
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
""")

conn.commit()
conn.close()

print("Success! attendance_logs table is ready.")