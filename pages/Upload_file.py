import os
import pickle
import re
import streamlit as st

from Main import splitting_doc, create_vector_storage, array_of_file_names_by_their_indexes


directory = "documents"
already_uploaded = False

# Если директории не существует, создадим ее
if not os.path.exists(directory):
    os.mkdir(directory)


# Функция для получения номера файла для его именования
def get_file_number():
    count_1 = 0
    count_2 = 0
    number_1 = ""
    for file_1 in os.listdir(directory):
        count_1 += 1
    for file_1 in os.listdir(directory):
        count_2 += 1
        if count_2 == count_1:
            number_1 = int(file_1.split("_")[0]) + 1
    if number_1 == "":
        number_1 = 1
    return number_1


# Выводим следующие сообщения пользователю
st.header("Здесь вы можете загрузить документы")
st.write("Внимание, загрузка файла может идти долго, и это нормально. "
         "Пожалуйста, не обновляйте страницу")


# Загрузка файлов
uploaded_file = st.file_uploader(label="Загрузите файл в формате txt", type='txt')
# Если загруженный файл не пуст
if uploaded_file is not None:
    # Получаем номер файла для именования
    number = get_file_number()
    file_name = ' '.join(uploaded_file.name.split())
    file_name = file_name.replace(" ", "_").replace(".txt", "")

    # Проверяем, нет ли уже файла с таким названием
    for file in os.listdir(directory):
        file = re.sub(r'\d_', lambda x: '', file, 1).replace(".pkl", "")
        if file in file_name:
            already_uploaded = True
    if already_uploaded:
        st.write("**Документ с таким названием уже загружен!**")

    # Если такого файла нет, загружаем его
    else:
        # Подготовка документа:
        try:
            uploaded_file = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            uploaded_file = uploaded_file.getvalue().decode("cp1251")

        split_doc = splitting_doc(uploaded_file)
        # Cоздание векторного хранилища на основе списка эмбеддингов:
        db = create_vector_storage(split_doc)

        file_name = f"{number}_{file_name}"

        # Сохраняем сериализованный объект в файл
        with open(f'documents/{file_name}.pkl', 'wb') as f:
            pickle.dump(db, f)

        st.write("**Документ успешно добавлен**")


# Функция для удаления файлов
def delete_files(files_array_1):
    # Проход по всем файлам в директории и удаление нужных
    for filename in os.listdir(directory):
        for name in files_array_1:
            if name == filename:
                os.remove(f"{directory}/{filename}")


# Смотрим, сколько у нас файлов в директории, и если они есть, выводим их и даем пользователю возможность удалять их
total_files = len([name for name in os.listdir(directory)])
if total_files == 0:
    st.write("**На данный момент загруженных файлов нет**")
else:
    # Вывод всех загруженных документов на экран
    text = ""
    for file in os.listdir(directory):
        text += "\n - >" + file
    st.write("**Загруженные документы:**")
    st.write(text)


    # Удаление файлов из загруженных
    # Выводим пользователю:
    st.write("Чтобы удалить документы из списка, введите ниже их номера через пробелы")
    files_numbers = st.text_input(label="Введите номера документов на удаление")
    # Создаем кнопку для подтверждения удаления
    file_num_but = st.button("Удалить выбранные докумнеты")

    # Если кнопка нажата, но ничего не введено
    if file_num_but and files_numbers == "":
        st.write("Сначала введите цифры в поле выше")

    # Условие, если кнопка нажата и что-то введено
    elif file_num_but and files_numbers != "":
        files_array = array_of_file_names_by_their_indexes(files_numbers, directory)
        # Если в строке встретилась не цифра
        if files_array == "It is not a number":
            st.write("Некорректный ввод, вводите целые числа документов через один пробел")
        # Если ни один документ не попал в массив
        elif len(files_array) == 0:
            st.write("Файлы с данными индексами не найдены")
        # Если какие-то документы попали в массив:
        else:
            st.write("Следующие файлы могут быть удалены:")
            st.write(files_array)
            # Если пользователь подтверждает удаление, удаляем файлы через функцию delete_files
            file_num_del = st.button("Удалить", on_click=delete_files(files_array))
            # Затем переименовываем все файлы:
            count = 1
            for file in os.listdir(directory):
                new_file_name = re.sub(r'\d_', lambda x: '_', file, 1)
                os.rename(f"{directory}/{file}", f"{directory}/{count}{new_file_name}")
                count += 1
