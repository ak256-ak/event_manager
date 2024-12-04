'''
from builtins import Exception, bool, classmethod, int, str
from datetime import datetime, timezone
import secrets
from typing import Optional, Dict, List
from pydantic import ValidationError
from sqlalchemy import func, null, update, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from app.dependencies import get_email_service, get_settings
from app.models.user_model import User
from app.schemas.user_schemas import UserCreate, UserUpdate
from app.utils.nickname_gen import generate_nickname
from app.utils.security import generate_verification_token, hash_password, verify_password
from uuid import UUID
from app.services.email_service import EmailService
from app.models.user_model import UserRole
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

class UserService:
    @classmethod
    async def _execute_query(cls, session: AsyncSession, query):
        try:
            result = await session.execute(query)
            await session.commit()
            return result
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            await session.rollback()
            return None

    @classmethod
    async def _fetch_user(cls, session: AsyncSession, **filters) -> Optional[User]:
        query = select(User).filter_by(**filters)
        result = await cls._execute_query(session, query)
        return result.scalars().first() if result else None

    @classmethod
    async def get_by_id(cls, session: AsyncSession, user_id: UUID) -> Optional[User]:
        return await cls._fetch_user(session, id=user_id)

    @classmethod
    async def get_by_nickname(cls, session: AsyncSession, nickname: str) -> Optional[User]:
        return await cls._fetch_user(session, nickname=nickname)

    @classmethod
    async def get_by_email(cls, session: AsyncSession, email: str) -> Optional[User]:
        return await cls._fetch_user(session, email=email)

    @classmethod
    async def create(cls, session: AsyncSession, user_data: Dict[str, str], email_service: EmailService) -> Optional[User]:
        try:
            validated_data = UserCreate(**user_data).model_dump()
            existing_user = await cls.get_by_email(session, validated_data['email'])
            if existing_user:
                logger.error("User with given email already exists.")
                return None
            validated_data['hashed_password'] = hash_password(validated_data.pop('password'))
            new_user = User(**validated_data)
            new_user.verification_token = generate_verification_token()
            new_nickname = generate_nickname()
            while await cls.get_by_nickname(session, new_nickname):
                new_nickname = generate_nickname()
            new_user.nickname = new_nickname
            session.add(new_user)
            await session.commit()
            await email_service.send_verification_email(new_user)
            
            return new_user
        except ValidationError as e:
            logger.error(f"Validation error during user creation: {e}")
            return None

    @classmethod
    async def update(cls, session: AsyncSession, user_id: UUID, update_data: Dict[str, str]) -> Optional[User]:
        try:
            # validated_data = UserUpdate(**update_data).dict(exclude_unset=True)
            validated_data = UserUpdate(**update_data).dict(exclude_unset=True)

            if 'password' in validated_data:
                validated_data['hashed_password'] = hash_password(validated_data.pop('password'))
            query = update(User).where(User.id == user_id).values(**validated_data).execution_options(synchronize_session="fetch")
            await cls._execute_query(session, query)
            updated_user = await cls.get_by_id(session, user_id)
            if updated_user:
                session.refresh(updated_user)  # Explicitly refresh the updated user object
                logger.info(f"User {user_id} updated successfully.")
                return updated_user
            else:
                logger.error(f"User {user_id} not found after update attempt.")
            return None
        except Exception as e:  # Broad exception handling for debugging
            logger.error(f"Error during user update: {e}")
            return None

    @classmethod
    async def delete(cls, session: AsyncSession, user_id: UUID) -> bool:
        user = await cls.get_by_id(session, user_id)
        if not user:
            logger.info(f"User with ID {user_id} not found.")
            return False
        await session.delete(user)
        await session.commit()
        return True

    @classmethod
    async def list_users(cls, session: AsyncSession, skip: int = 0, limit: int = 10) -> List[User]:
        query = select(User).offset(skip).limit(limit)
        result = await cls._execute_query(session, query)
        return result.scalars().all() if result else []

    @classmethod
    async def register_user(cls, session: AsyncSession, user_data: Dict[str, str], get_email_service) -> Optional[User]:
        return await cls.create(session, user_data, get_email_service)
    

    @classmethod
    async def login_user(cls, session: AsyncSession, email: str, password: str) -> Optional[User]:
        user = await cls.get_by_email(session, email)
        if user:
            if user.email_verified is False:
                return None
            if user.is_locked:
                return None
            if verify_password(password, user.hashed_password):
                user.failed_login_attempts = 0
                user.last_login_at = datetime.now(timezone.utc)
                session.add(user)
                await session.commit()
                return user
            else:
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= settings.max_login_attempts:
                    user.is_locked = True
                session.add(user)
                await session.commit()
        return None

    @classmethod
    async def is_account_locked(cls, session: AsyncSession, email: str) -> bool:
        user = await cls.get_by_email(session, email)
        return user.is_locked if user else False


    @classmethod
    async def reset_password(cls, session: AsyncSession, user_id: UUID, new_password: str) -> bool:
        hashed_password = hash_password(new_password)
        user = await cls.get_by_id(session, user_id)
        if user:
            user.hashed_password = hashed_password
            user.failed_login_attempts = 0  # Resetting failed login attempts
            user.is_locked = False  # Unlocking the user account, if locked
            session.add(user)
            await session.commit()
            return True
        return False

    @classmethod
    async def verify_email_with_token(cls, session: AsyncSession, user_id: UUID, token: str) -> bool:
        user = await cls.get_by_id(session, user_id)
        if user and user.verification_token == token:
            user.email_verified = True
            user.verification_token = None  # Clear the token once used
            user.role = UserRole.AUTHENTICATED
            session.add(user)
            await session.commit()
            return True
        return False

    @classmethod
    async def count(cls, session: AsyncSession) -> int:
        """
        Count the number of users in the database.

        :param session: The AsyncSession instance for database access.
        :return: The count of users.
        """
        query = select(func.count()).select_from(User)
        result = await session.execute(query)
        count = result.scalar()
        return count
    
    @classmethod
    async def unlock_user_account(cls, session: AsyncSession, user_id: UUID) -> bool:
        user = await cls.get_by_id(session, user_id)
        if user and user.is_locked:
            user.is_locked = False
            user.failed_login_attempts = 0  # Optionally reset failed login attempts
            session.add(user)
            await session.commit()
            return True
        return False
'''
import pytest
from unittest.mock import MagicMock, AsyncMock
from app.services.user_service import UserService
from app.models.user_model import User, UserRole
from app.utils.security import hash_password
from uuid import uuid4

@pytest.fixture
def mock_session():
    """Mock the AsyncSession."""
    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()
    session.add = MagicMock()
    session.delete = AsyncMock()
    return session


@pytest.fixture
def mock_user():
    """Mock user object."""
    return User(
        id=uuid4(),
        email="test@example.com",
        nickname="testuser",
        hashed_password=hash_password("password123"),
        role=UserRole.AUTHENTICATED,
        email_verified=True
    )


@pytest.fixture
def mock_email_service():
    """Mock the email service."""
    email_service = MagicMock()
    email_service.send_verification_email = AsyncMock()
    return email_service


@pytest.mark.asyncio
async def test_create_user_success(mock_session, mock_email_service):
    """Test successful user creation."""
    user_data = {
        "email": "newuser@example.com",
        "password": "password123",
        "first_name": "New",
        "last_name": "User"
    }

    result = await UserService.create(mock_session, user_data, mock_email_service)

    assert result is not None
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    mock_email_service.send_verification_email.assert_called_once()


@pytest.mark.asyncio
async def test_create_user_duplicate_email(mock_session, mock_email_service, mock_user):
    """Test user creation with an existing email."""
    mock_session.execute.return_value.scalars.return_value.first.return_value = mock_user

    user_data = {
        "email": mock_user.email,
        "password": "password123",
    }

    result = await UserService.create(mock_session, user_data, mock_email_service)

    assert result is None
    mock_email_service.send_verification_email.assert_not_called()
    mock_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_update_user_success(mock_session, mock_user):
    """Test successful user update."""
    mock_session.execute.return_value.scalars.return_value.first.return_value = mock_user

    update_data = {
        "first_name": "Updated",
        "last_name": "Name",
    }

    result = await UserService.update(mock_session, mock_user.id, update_data)

    assert result is not None
    assert result.first_name == "Updated"
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_update_user_not_found(mock_session):
    """Test user update when the user is not found."""
    mock_session.execute.return_value.scalars.return_value.first.return_value = None

    update_data = {
        "first_name": "Updated",
    }

    result = await UserService.update(mock_session, uuid4(), update_data)

    assert result is None
    mock_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_delete_user_success(mock_session, mock_user):
    """Test successful user deletion."""
    mock_session.execute.return_value.scalars.return_value.first.return_value = mock_user

    result = await UserService.delete(mock_session, mock_user.id)

    assert result is True
    mock_session.commit.assert_called_once()
    mock_session.delete.assert_called_once_with(mock_user)


@pytest.mark.asyncio
async def test_delete_user_not_found(mock_session):
    """Test user deletion when the user is not found."""
    mock_session.execute.return_value.scalars.return_value.first.return_value = None

    result = await UserService.delete(mock_session, uuid4())

    assert result is False
    mock_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_login_user_success(mock_session, mock_user):
    """Test successful user login."""
    mock_session.execute.return_value.scalars.return_value.first.return_value = mock_user

    result = await UserService.login_user(mock_session, mock_user.email, "password123")

    assert result is not None
    assert result.email == mock_user.email
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_login_user_invalid_credentials(mock_session, mock_user):
    """Test login with invalid credentials."""
    mock_session.execute.return_value.scalars.return_value.first.return_value = mock_user

    result = await UserService.login_user(mock_session, mock_user.email, "wrongpassword")

    assert result is None
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_list_users(mock_session, mock_user):
    """Test listing users."""
    mock_session.execute.return_value.scalars.return_value.all.return_value = [mock_user]

    result = await UserService.list_users(mock_session, skip=0, limit=10)

    assert len(result) == 1
    assert result[0].email == mock_user.email


@pytest.mark.asyncio
async def test_verify_email_with_token_success(mock_session, mock_user):
    """Test successful email verification."""
    mock_user.verification_token = "valid_token"
    mock_session.execute.return_value.scalars.return_value.first.return_value = mock_user

    result = await UserService.verify_email_with_token(mock_session, mock_user.id, "valid_token")

    assert result is True
    assert mock_user.email_verified is True
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_verify_email_with_token_invalid(mock_session, mock_user):
    """Test email verification with an invalid token."""
    mock_user.verification_token = "valid_token"
    mock_session.execute.return_value.scalars.return_value.first.return_value = mock_user

    result = await UserService.verify_email_with_token(mock_session, mock_user.id, "invalid_token")

    assert result is False
    mock_session.commit.assert_not_called()
