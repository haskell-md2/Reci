import ALSmanager
import pandas as pd
import os
from fastapi import FastAPI, HTTPException

from catboost import CatBoostClassifier

import re

# Константы
MODEL_PATH = "/models/als_ma"
CAT_PATH = "/models/rec_cat.ctb"
DATA_DIR = "dataset"
MY_USER_ID = 159614
RETEACH = False

# Инициализация FastAPI
app = FastAPI()

# Глобальные переменные (инициализируются при старте)
als_manager = None
items_df = None

def initialize_als_manager():
    """Инициализация ALS менеджера и загрузка данных"""
    global als_manager, items_df, users_df, cat_bst
    
    try:
        # Получение путей
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(current_dir)
        full_model_path = os.path.join(current_dir, MODEL_PATH.lstrip('/'))
        data_dir_path = os.path.join(parent_dir, DATA_DIR)
        
        # Загрузка данных
        interactions_path = os.path.join(data_dir_path, "interactions.csv")
        items_path = os.path.join(data_dir_path, "items.csv")
        users_path = os.path.join(data_dir_path, "users.csv")
        
        inter = pd.read_csv(interactions_path)
        items_df = pd.read_csv(items_path)
        users_df = pd.read_csv(users_path)
        
        # Инициализация модели
        if RETEACH:
            als_manager = ALSmanager.ALSmanager(path_to_model=full_model_path, inter=inter)
        else:
            als_manager = ALSmanager.ALSmanager(path_to_model=full_model_path, inter=inter, new=False)
        
        # Добавление пользовательских оценок
        user_ratings = [
            (MY_USER_ID, 95084, 0, 5, '2019-12-31'),
            (MY_USER_ID, 136190, 0, 5, '2019-12-31'),
            (MY_USER_ID, 215257, 0, 3, '2019-12-31'),
            (MY_USER_ID, 182602, 0, 5, '2019-12-31'),
            (MY_USER_ID, 265255, 0, 4, '2019-12-31'),
            (MY_USER_ID, 126875, 0, 4, '2019-12-31'),
            (MY_USER_ID, 126581, 0, 3, '2019-12-31'),
            (MY_USER_ID, 63201, 0, 2, '2019-12-31'),
            (MY_USER_ID, 242075, 0, 5, '2019-12-31'),
            (MY_USER_ID, 293034, 0, 1, '2019-12-31'),
            (MY_USER_ID, 209379, 0, 2, '2019-12-31')
        ]
        
        # Добавляем оценки в DataFrame
        for rating in user_ratings:
            inter.loc[len(inter)] = rating
        
        # Обновляем модель
        als_manager.update_users([MY_USER_ID], inter)
        
        cat_model_path = os.path.join(current_dir, CAT_PATH.lstrip('/'))
        print(cat_model_path)

        cat_bst = CatBoostClassifier()
        cat_bst.load_model(cat_model_path)

        print("ALS manager initialized successfully")
        
    except Exception as e:
        print(f"Error initializing ALS manager: {e}")
        raise

def get_titles_by_ids(book_ids, items_data):
    """Получение названий книг по их ID"""
    try:
        titles = []
        for book_id in book_ids:


            book_title = items_data[items_data['id'] == book_id]['title']
            if not book_title.empty:
                titles.append(book_title.values[0])
            else:
                titles.append(f"Unknown book (ID: {book_id})")
        return titles
    except Exception as e:
        print(f"Error getting titles: {e}")
        return []

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    initialize_als_manager()

@app.get("/")
async def read_root():
    """Главная страница с рекомендациями для основного пользователя"""
    try:
        if als_manager is None or items_df is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        recommended_ids = als_manager.predict_for_user(MY_USER_ID, 100)
        recommended_books = get_titles_by_ids(recommended_ids, items_df)
        
        return {
            "user_id": MY_USER_ID,
            "recommendations_count": len(recommended_books),
            "recommendations": recommended_books
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


def prepare_features(user_id, book_ids, users_df, items_df):
    pass

@app.get("/recommend/{user_id}")
async def recommend_for_user(user_id: int, count: int = 100):
    """Рекомендации для конкретного пользователя"""
    try:
        if als_manager is None or items_df is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if count <= 0 or count > 1000:
            raise HTTPException(status_code=400, detail="Count must be between 1 and 1000")
        
        recommended_ids = als_manager.predict_for_user(user_id, count)

        #Формируем параметры.

        X_new = prepare_features(user_id, recommended_ids, users_df, items_df)

        res = cat_bst.predict_proba(X_new)

        print(res)

        recommended_books = get_titles_by_ids(recommended_ids, items_df)
        
        
        return {
            "user_id": user_id,
            "recommendations_count": len(recommended_books),
            "recommendations": recommended_books
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid user ID: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

if __name__ == '__main__':
    # Для тестирования при прямом запуске
    initialize_als_manager()
    
    # Тест рекомендаций
    test_ids = als_manager.predict_for_user(MY_USER_ID, 5)
    test_books = get_titles_by_ids(test_ids, items_df)
    
    print("Тестовые рекомендации:")
    for i, book in enumerate(test_books, 1):
        print(f"{i}. {book}")
    
    print("\nДля запуска сервера выполните:")
    print("uvicorn script_name:app --reload")