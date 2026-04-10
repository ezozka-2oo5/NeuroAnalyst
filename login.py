# login.py - TO'G'RILANGAN VERSIYA
import json
import os
import hashlib

class LoginSystem:
    def __init__(self, users_file='users.json'):
        self.users_file = users_file
        self.current_user = None
        self.load_users()
    
    def load_users(self):
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        else:
            # Yangi foydalanuvchilar yaratish
            self.users = {
                "admin": {
                    "password": self.hash_password("admin123"),
                    "role": "admin",
                    "name": "Administrator"
                },
                "doctor": {
                    "password": self.hash_password("doctor123"),
                    "role": "doctor",
                    "name": "Shifokor"
                }
            }
            self.save_users()
    
    def save_users(self):
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=4)
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def login(self, username, password):
        if username in self.users:
            # Kiritilgan parolni hash'lab, saqlangan bilan solishtirish
            hashed_input = self.hash_password(password)
            if self.users[username]["password"] == hashed_input:
                self.current_user = {
                    "username": username,
                    "role": self.users[username]["role"],
                    "name": self.users[username]["name"]
                }
                return True, "Muvaffaqiyatli kirish"
            else:
                print(f"Noto'g'ri: {hashed_input} != {self.users[username]['password']}")
                return False, "Noto'g'ri parol"
        return False, "Foydalanuvchi topilmadi"