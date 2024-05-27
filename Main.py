import os  # Модуль для взаимодействия с операционной системой, такой как доступ к переменным окружения
from getpass import getpass  # Импорт функции для скрытого ввода паролей или конфиденциальных данных
import pickle  # "Pickling" - процесс преобразования объекта Python в поток байтов, а "unpickling"
# - обратная операция, в результате которой поток байтов преобразуется обратно в Python-объект.
import streamlit as st  # Streamlit — это фреймворк для языка программирования Python. Он содержит
# набор программных инструментов, которые помогают перенести модель машинного обучения в веб
import re  # для регулярных выражений
from dotenv import load_dotenv  # Python-dotenv считывает пары ключ-значение из файла
# .env и может устанавливать их как переменные среды
# from tqdm import tqdm # для засекания времени в консоли


# Для сбора компонентов RAG:
from langchain_community.document_loaders import Docx2txtLoader  # Импорт класса для загрузки документов формата .docx
from langchain_community.document_loaders import TextLoader  # Импорт класса для загрузки текстовых документов
from langchain.text_splitter import (
    CharacterTextSplitter,  # Отвечает за разделение текста на отдельные символы
    RecursiveCharacterTextSplitter,  # Используется для более сложной рекурсивной обработки текста
)
from langchain_community.embeddings import HuggingFaceEmbeddings  # Предобученные эмбеддинги
# (векторные представления) текста
from langchain_community.vectorstores import FAISS  # Библиотека, предоставляющая эффективные
# алгоритмы для быстрого поиска и кластеризации эмбеддингов
from langchain_community.retrievers import BM25Retriever  # , EnsembleRetriever

# Для подключения RAG к LLM:
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser

import warnings  # Модуль для управления предупреждениями

warnings.filterwarnings("ignore")


# Установка переменной окружения
def set_environ():
    load_dotenv()
    # os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'Введите ваш HuggingFaceHub API ключ'
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def uploading_file(path: str):
    with open(path, encoding="cp1251", errors='ignore') as f:
        doc_1 = f.read()
    return doc_1


def splitting_doc(doc_1):
    # Определяем сплиттер:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    # Разбиваем документ:
    split_doc_1 = splitter.create_documents([doc_1])
    return split_doc_1


def create_vector_storage(split_doc_1):
    # Задаём embedding model
    # Если у вас нет видеокарты, укажите 'device': 'cpu'
    hf_embeddings_model = HuggingFaceEmbeddings(
        model_name="cointegrated/LaBSE-en-ru",
        model_kwargs={"device": "cpu"}
    )

    # Создаем FAISS индекс (базу векторов) с полученными эмбеддингами
    db_1 = FAISS.from_documents(split_doc_1, hf_embeddings_model)
    return db_1


def create_retriever(my_db_1):
    retriever_1 = my_db_1.as_retriever()
    return retriever_1


def remove_indentation(text):
    # Разбили текст по переносом строк:
    lines = text.split('\n')
    # Левое обрезание лишних пробелов для каждого элмента:
    trimmed_lines = [line.lstrip() for line in lines]
    # Снова собрали в одну строку:
    return '\n'.join(trimmed_lines)


# Объявляем функцию, которая будет собирать строку из полученных документов
def format_docs(docs):
    # Собираем строку из полученных документов-ответов
    string = "\n".join([d.page_content.strip() for d in docs])
    # Убираем лишние отступы после каждого переноса строки и снова формируем одну строку
    # (это чтобы не было табов, которые streamlit распознает как строку кода)
    string = remove_indentation(string)
    string = re.sub(r'\n(\d+)\.\d\.\s', lambda x: f'\n\n---\n **-> {x.group().rstrip()}** ', string)
    string = re.sub(r'\n(\d+)\.\s', lambda x: f'\n\n---\n **-> {x.group().rstrip()}** ', string)
    string = string.replace(':', ':\n').replace(';', ';\n')
    string = re.sub(r'[\s][А-Яа-я]\)', lambda x: f'\n\n {x.group()}', string)
    #string = "\n".join([line.capitalize() for line in string.split("\n")])
    #string = re.sub(r'[А-Я]\)', lambda x: f'{x.group().lower()}', string)
    return string


def connection_Mixtral(retriever_1):
    # Инииализация модели с HF
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )

    # Создаём простой шаблон
    template = """
    **Question:** {question}\n\n**Answer the question based only on 
    the following context:**\n\n{context}\n\n"""

    # Создаём промпт из шаблона
    prompt = PromptTemplate(template=template, input_variables=["question"])

    return (
            {"context": retriever_1 | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )


def get_current_files_names(directory_1):
    files_names = ""
    for f in os.listdir(directory_1):
        files_names += "\n" + f
    return files_names + "\n\n"


def array_of_file_names_by_their_indexes(files_numbers_1, directory_1):
    files_numbers_1 = files_numbers_1.split(" ")

    # Если в полученном массиве не только числа:
    for num in files_numbers_1:
        if not num.isdigit():
            return "It is not a number"

    # Иначе вытаскиваем индексы, файлы которых у нас есть:
    files_names_array_1 = []
    for f in os.listdir(directory_1):
        file_index = f.split("_")[0]
        for num in files_numbers_1:
            if int(num) == int(file_index):
                files_names_array_1.append(f)
    return files_names_array_1


# Соберём компоненты RAG по порядку
def to_collect_RAG(files_names_array_1):
    # Собираем все файлы в один:
    count = 0
    # Проход по всем файлам в директории
    for filename in files_names_array_1:
        if count == 0:
            count += 1
            # Загружаем объект db из файла:
            with open(f"documents/{filename}", 'rb') as file:
                db = pickle.load(file)
        else:
            # Добавляем файлы в db:
            with open(f"documents/{filename}", 'rb') as file:
                piece_db = pickle.load(file)
                db.merge_from(piece_db)

    # Задаём ретривер:
    retriever = create_retriever(db)
    # Подключаем Mixtral и создаём цепочку:
    chain = connection_Mixtral(retriever)
    return chain


def load_question():
    uploaded_question = st.text_input(label="Напишите вопрос")
    if uploaded_question is not None:
        return uploaded_question
    else:
        return None


def get_answer(question_1: str, files_names_array_1):
    chain = to_collect_RAG(files_names_array_1)
    return chain.invoke(question_1)


if __name__ == '__main__':
    set_environ()
    directory = "documents/"

    if not os.path.exists(directory):
        os.mkdir(directory)

    # Подключаем Streamlit
    # Выводим сообщения ниже пользователю:
    st.set_page_config("Welcome")
    st.sidebar.success("Выберите вкладку здесь")
    st.title("Ответы на вопросы по нормативным документам. Языковая модель Mixtral")
    st.write("**Инструкция по использованию:** \n"
             "1) Загрузите свои документы в соседней вкладке\n"
             "2) Выберите номера документов для формирования ответа\n"
             "3) Напишите вопрос")

    # Считаем, сколько всего загруженных файлов в директории
    total_files = len([name for name in os.listdir(directory)])
    # Если загруженных файлов нет, просим пользователя сначала загрузить что-то
    if total_files == 0:
        st.write("**Для начала работы загрузите хотя бы один файл в соседней вкладке**")
    # А если загруженные файлы есть
    else:
        st.write(f"**Доступные документы**:\n{get_current_files_names(directory)}\n*Оставьте поле ниже пустым и "
                 "при ответе будут использованы все файлы*")
        # Создаем поле для ввода номеров документов:
        files_numbers = st.text_input(label="Через пробел напишите цифры документов, которые хотите использовать "
                                            "при получении ответа, затем нажмите кнопку 'Использовать эти документы'")
        # Создаем кнопку:
        files_num_but = st.button("Использовать эти документы")


        # Теперь сформируем массив названий документов, которые будет использовать для получения ответа
        # Если кнопка нажата, но ничего не введено
        if files_num_but and files_numbers == "":
            st.write("Сначала введите цифры в поле выше")


        # Условие, если кнопка нажата и что-то введено
        if files_num_but and files_numbers != "":
            # Проверяем, что ввел пользователь и собираем массив названий нужных документов:
            files_names_array = array_of_file_names_by_their_indexes(files_numbers, directory)
            # Если в строке встретилась не цифра
            if files_names_array == "It is not a number":
                st.write("Некорректный ввод, вводите целые числа документов через один пробел")
            # Если ни один документ не попал в массив
            elif len(files_names_array) == 0:
                st.write("Не найдены файлы с данными индексами, будут использованы все документы")
                for files in os.listdir(directory):
                    files_names_array.append(files)
            # Если какие-то документы попали в массив, будем использовать их для ответа
            else:
                st.write("В ответе будут использоваться документы:")
                st.write(files_names_array)


        # Если пользователь не воспользовался полем и кнопкой
        else:
            files_names_array = []
            # Все файлы директории будут использованы для получения ответа:
            for files in os.listdir(directory):
                files_names_array.append(files)


        # Просим пользователя ввести вопрос, затем формируем и печатаем ответ на него
        st.write("\n\nВведите вопрос ниже")
        question = load_question()
        result = st.button("Получить ответ")
        if result and question == "":
            st.write("Напишите вопрос!")
        elif result:
            context = get_answer(question, files_names_array)
            st.write(context)


# Какие требования к обеспечению безопасности в ходе создания, эксплуатации и вывода из
# эксплуатации значимых объектов нужно выполнять?
