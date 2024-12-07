from fastapi import HTTPException
from ..models.user_model import UserModel


class UserService:
    def __init__(self):
        self.users_db = []

    def create_user(self, user: UserModel) -> UserModel:
        self.users_db.append(user)
        return user

    def get_users(self) -> list:
        return self.users_db

    def get_user(self, user_id: int) -> UserModel:
        for user in self.users_db:
            if user.id == user_id:
                return user
        raise HTTPException(status_code=404, detail="User not found")

    def update_user(self, user_id: int, updated_user: UserModel) -> UserModel:
        for index, user in enumerate(self.users_db):
            if user.id == user_id:
                self.users_db[index] = updated_user
                return updated_user
        raise HTTPException(status_code=404, detail="User not found")

    def delete_user(self, user_id: int) -> UserModel:
        for index, user in enumerate(self.users_db):
            if user.id == user_id:
                return self.users_db.pop(index)
        raise HTTPException(status_code=404, detail="User not found")
