from fastapi import APIRouter
from ..models.user_model import UserModel
from ..services.user_services import UserService


class UserRouter:
    def __init__(self):
        self.router = APIRouter()
        self.user_service = UserService()

        self.router.post("/users/", response_model=UserModel)(self.create_user_route)
        self.router.get("/users/", response_model=list)(self.get_users_route)
        self.router.get("/users/{user_id}", response_model=UserModel)(
            self.get_user_route
        )
        self.router.put("/users/{user_id}", response_model=UserModel)(
            self.update_user_route
        )
        self.router.delete("/users/{user_id}", response_model=UserModel)(
            self.delete_user_route
        )

    def create_user_route(self, user: UserModel):
        return self.user_service.create_user(user)

    def get_users_route(self):
        return self.user_service.get_users()

    def get_user_route(self, user_id: int):
        return self.user_service.get_user(user_id)

    def update_user_route(self, user_id: int, updated_user: UserModel):
        return self.user_service.update_user(user_id, updated_user)

    def delete_user_route(self, user_id: int):
        return self.user_service.delete_user(user_id)


# Create an instance of UserRouter to use in main.py
user_router = UserRouter().router
