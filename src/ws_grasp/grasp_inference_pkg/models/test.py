import torch

model_path = "model_5700.pth"

try:
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    print("✅ Файл загружен успешно!")
    print(f"Тип данных: {type(checkpoint)}")
    
    # Посмотреть структуру
    if isinstance(checkpoint, dict):
        print(f"Ключи в checkpoint: {list(checkpoint.keys())}")
        if 'model_state_dict' in checkpoint:
            print(f"Количество слоев: {len(checkpoint['model_state_dict'])}")
    
except Exception as e:
    print(f"❌ Ошибка при загрузке: {e}")



import zipfile
import os

def check_checkpoint(model_path):
    # Проверка существования и размера
    if not os.path.exists(model_path):
        print(f"❌ Файл не найден: {model_path}")
        return False
    
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📦 Размер файла: {size_mb:.2f} MB")
    
    if size_mb == 0:
        print("❌ Файл пустой!")
        return False
    
    # Проверка ZIP-архива
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_file:
            files = zip_file.namelist()
            print(f"✅ ZIP-архив корректный")
            print(f"📄 Файлы внутри ({len(files)}): {files[:5]}...")  # Первые 5
            
            # Проверка на битые файлы внутри
            bad_files = zip_file.testzip()
            if bad_files:
                print(f"❌ Поврежденные файлы: {bad_files}")
                return False
            else:
                print("✅ Все файлы внутри архива целые")
            
            return True
    except zipfile.BadZipFile:
        print("❌ Файл поврежден - не является корректным ZIP")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

# Использование
check_checkpoint(model_path)