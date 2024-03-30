import json
import logging

import telebot
from telebot.types import Message

import db
from config import ADMINS, LOGS_PATH, MAX_TOKENS_PER_SESSION, MAX_SESSIONS, MAX_USERS, MAX_MODEL_TOKENS, BOT_TOKEN
from gpt import ask_gpt_helper, count_tokens_in_dialogue, get_system_content
from utils import create_keyboard

# Инициируем логгер по пути константы с уровнем логгирования debug
logging.basicConfig(
    filename=LOGS_PATH,
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
    filemode="w",
)

# Создаем клиент нашего бота
bot = telebot.TeleBot(BOT_TOKEN)

# Создаем базу и табличку в ней
db.create_db()
db.create_table()

# Определяем списки предметов и уровней сложности
subjects_list = [
    "Хоррор",
    "Юмор",
    "Фэнтези",
]
levels_list = ["Простой", "Уголоблённый", "Ещё более углублённый"]


@bot.message_handler(commands=["start"])
def start(message):
    user_name = message.from_user.first_name  # Получаем имя пользователя
    user_id = message.from_user.id  # Получаем id пользователя

    if not db.is_user_in_db(user_id):  # Если пользователя в базе нет
        if len(db.get_all_users_data()) < MAX_USERS:  # Если число зарегистрированных пользователей меньше допустимого
            db.add_new_user(user_id)  # Регистрируем нового пользователя
        else:
            # Если уперлись в лимит пользователей, отправляем соответствующее письмо
            bot.send_message(
                user_id,
                "К сожалению, лимит пользователей исчерпан. "
                "Вы не сможете воспользоваться ботом:("
            )
            return  # Прерываем здесь функцию, чтобы дальнейший код не выполнялся

    # Этот блок срабатывает только для зарегистрированных пользователей
    bot.send_message(
        user_id,
        f"Привет, {user_name}! Я бот-сценарист сочиняющий разные жанры\n"
        f"Ты можешь выбрать жанр и углублённость сюжета, написать желаемого героя, а я сочиню сценарий.\n"
        f"Ты всегда можешь попросить продолжить сценарий",
        reply_markup=create_keyboard(["Выбрать жанр"]),  # Добавляем кнопочку для ответа
    )
    # Насильно уводим пользователя в функцию выбора предмета (независимо от того нажмет ли он кнопку или отправит
    # какое-нибудь другое сообщение)
    bot.register_next_step_handler(message, choose_subject)


# Функция фильтр для хэндлера выбора предмета (создания новой сессии)
def filter_choose_subject(message: Message) -> bool:
    user_id = message.from_user.id
    if db.is_user_in_db(user_id):  # Отработает только для зарегистрированных пользователей
        return message.text in ["Выбрать жанр", "Выбрать другой жанр", "Создать новый сценарий"]
    return False


@bot.message_handler(func=filter_choose_subject)
def choose_subject(message: Message):
    user_id = message.from_user.id
    sessions = db.get_user_data(user_id)["sessions"]  # Получаем из БД актуальное количество сессий пользователя
    if sessions < MAX_SESSIONS:  # Если число сессий пользователя не достигло предела
        db.update_row(user_id, "sessions", sessions + 1)  # Накручиваем ему +1 сессию
        db.update_row(user_id, "tokens", MAX_TOKENS_PER_SESSION)  # И обновляем его токены
        bot.send_message(
            user_id,
            "Выбери жанр, по которому хочешь написать сценарий:",
            reply_markup=create_keyboard(subjects_list),  # Создаем клавиатуру из списка предметов
        )
        bot.register_next_step_handler(message, subject_selection)  # Принудительно уводим в функцию выбора сложности

    else:  # Если число сессий достигло лимита
        bot.send_message(
            user_id,
            "К сожалению, лимит твоих токенов исчерпан:("
        )


def subject_selection(message: Message):
    user_id = message.from_user.id
    user_choice = message.text  # Получаем выбор предмета пользователя
    if user_choice in subjects_list:  # Проверим, что предмет есть в списке. Это исключит вариант, если пользователь
        # захочет ввести собственный предмет, вместо того, чтобы выбрать из кнопок на клавиатуре
        db.update_row(user_id, "subject", user_choice)  # Обновим значение предмета в БД
        bot.send_message(
            user_id,
            f"Отлично, {message.from_user.first_name}, теперь я буду сочинять тебе сценарий по жанру '{user_choice}'!"
            f"Давай теперь выберем углоблённость сюжета по этому жанру.",
            reply_markup=create_keyboard(levels_list),  # Создаем клавиатуру выбора сложности из списка
        )
        bot.register_next_step_handler(message, level_selection)  # Принудительно уводим в обработчик выбора сложности

    else:  # Если был выбран предмет не из нашего списка
        bot.send_message(
            user_id,
            "К сожалению, по такому жанру я не смогу тебе сочинить сценарий, выбери один из предложенных в меню",
            reply_markup=create_keyboard(subjects_list),
        )
        bot.register_next_step_handler(message, subject_selection)  # Снова отправляем его в эту же функцию


def filter_choose_level(message: Message) -> bool:
    user_id = message.from_user.id
    if db.is_user_in_db(user_id):
        return message.text == "Изменить углублённость сюжета"


# Выбор уровня сложности построен аналогично выбору предмета, поэтому тут не расписываю
@bot.message_handler(func=filter_choose_level)
def choose_level(message: Message):
    bot.send_message(
        message.from_user.id,
        "Какой уровень углублённости сюжета тебе нужен:",
        reply_markup=create_keyboard(levels_list),
    )
    bot.register_next_step_handler(message, level_selection)


def level_selection(message: Message):
    user_id = message.from_user.id
    user_choice = message.text
    if user_choice in levels_list:
        db.update_row(user_id, "level", user_choice)
        bot.send_message(
            user_id,
            f"Принято, {message.from_user.first_name}! Теперь мои сценарии будут углублённости: '{user_choice}'. "
            f"А теперь расскажи мне свои пожелания",
        )
        bot.register_next_step_handler(message, give_answer)
    else:
        bot.send_message(
            user_id,
            "Пожалуйста, выбери углублённость из предложенных:",
            reply_markup=create_keyboard(levels_list),
        )
        bot.register_next_step_handler(message, level_selection)


def filter_solve_task(message: Message) -> bool:
    user_id = message.from_user.id
    if db.is_user_in_db(user_id):
        return message.text == "Создать новый сценарий"


# Блок решения задач
@bot.message_handler(func=filter_solve_task)
def solve_task(message):
    bot.send_message(message.from_user.id, "Напиши условие задачи:")  # Просим пользователя ввести вопрос
    bot.register_next_step_handler(message, give_answer)  # И принудительно уводим в функцию предоставления ответа


def give_answer(message: Message):
    user_id = message.from_user.id
    user_tokens = db.get_user_data(user_id)["tokens"]  # Получаем актуальное количество токенов пользователя из БД
    subject = db.get_user_data(user_id)["subject"]  # Получаем выбранный предмет из БД
    level = db.get_user_data(user_id)["level"]  # Получаем выбранную сложность из БД

    user_content = message.text  # Формируем user_content из сообщения пользователя
    system_content = get_system_content(subject, level)  # Формируем system_content из предмета и сложности

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]  # Приводим контент к стандартизированному виду - списку из словарей сообщений
    tokens_messages = count_tokens_in_dialogue(messages)  # Посчитаем вес запроса в токенах

    if tokens_messages + MAX_MODEL_TOKENS <= user_tokens:  # Проверим что вес запроса + максимального ответа меньше, чем
        # оставшееся количество токенов у пользователя, чтобы пользователю хватило и на запрос и на максимальный ответ
        bot.send_message(message.from_user.id, "Решаю...")
        answer = ask_gpt_helper(messages)  # Получаем ответ от GPT
        messages.append({"role": "assistant", "content": answer})  # Добавляем в наш словарик ответ GPT

        user_tokens -= count_tokens_in_dialogue([{"role": "assistant", "content": answer}])  # Получаем новое значение
        # оставшихся токенов пользователя - вычитаем стоимость запроса и ответа
        db.update_row(user_id, "tokens", user_tokens)  # Записываем новое значение в БД

        json_string = json.dumps(messages, ensure_ascii=False)  # Преобразуем список словарей сообщений к виду json
        # строки для хранения в одной ячейке БД
        db.update_row(user_id, "messages", json_string)  # Записываем получившуюся строку со всеми
        # сообщениями в ячейку 'messages'

        if answer is None:
            bot.send_message(
                user_id,
                "Не могу получить ответ от АгенстваGPT",
                reply_markup=create_keyboard(
                    [
                        "Создать новый сценарий",
                        "Выбрать другой жанр",
                        "Изменить углублённость сюжета",
                    ]
                ),
            )
        elif answer == "":
            bot.send_message(
                user_id,
                "Не могу создать сценарий >:(",
                reply_markup=create_keyboard(
                    [
                        "Создать новый сценарий",
                        "Выбрать другой жанр",
                        "Изменить углублённость сюжета",
                    ]
                ),
            )
            logging.info(
                f"Отправлено: {message.text}\nПолучена ошибка: нейросеть вернула пустую строку"
            )
        else:
            bot.send_message(
                user_id,
                answer,
                reply_markup=create_keyboard(
                    [
                        "Создать новый сценарий",
                        "Продолжить сценарий",
                        "Выбрать другой жанр",
                        "Изменить углублённость сюжета",
                    ]
                ),
            )

    else:  # Если у пользователя не хватает токенов на запрос + ответ
        bot.send_message(
            message.from_user.id,
            "Токенов на ответ может не хватить:( Начни новую сессию",
            reply_markup=create_keyboard(["Начать новую сессию"])  # Предлагаем ему начать новую сессию через кнопку
        )
        logging.info(
            f"Отправлено: {message.text}\nПолучено: Предупреждение о нехватке токенов"
        )


def filter_continue_explaining(message: Message) -> bool:
    user_id = message.from_user.id
    if db.is_user_in_db(user_id):
        return message.text == "Продолжить сценарий"


@bot.message_handler(func=filter_continue_explaining)
def continue_explaining(message):
    user_id = message.from_user.id
    json_string_messages = db.get_user_data(user_id)["messages"]  # Достаем из базы все предыдущие сообщения
    # в виде json-строки
    messages = json.loads(json_string_messages)  # Преобразуем json-строку в нужный нам формат списка словарей
    if not messages:  # Если попытались продолжить, но запроса еще не было
        bot.send_message(
            user_id,
            "Для начала напиши свои пожелания:",
            reply_markup=create_keyboard(["Создать новый сценарий"]),
        )
        return  # Прерываем выполнение функции

    user_tokens = db.get_user_data(user_id)["tokens"]  # Получаем актуальное количество токенов пользователя
    tokens_messages = count_tokens_in_dialogue(messages)  # Считаем вес запроса в токенах из всех предыдущих сообщений

    if tokens_messages + MAX_MODEL_TOKENS <= user_tokens:  # Проверяем хватает ли токенов на запрос + ответ
        bot.send_message(user_id, "Формулирую продолжение...")
        answer = ask_gpt_helper(messages)  # Получаем продолжение от gpt
        messages.append({"role": "assistant", "content": answer})  # Добавляем очередной ответ в список сообщений

        user_tokens -= count_tokens_in_dialogue([{"role": "assistant", "content": answer}])  # Вычитаем токены
        db.update_row(user_id, "tokens", user_tokens)  # Сохраняем новое значение токенов в БД

        json_string_messages = json.dumps(messages, ensure_ascii=False)  # Преобразуем список сообщений в строку для БД
        db.update_row(user_id, "messages", json_string_messages)  # Сохраняем строку сообщений в БД

        if answer is None:
            bot.send_message(
                user_id,
                "Не могу получить ответ от АгенстваGPT",
                reply_markup=create_keyboard(
                    [
                        "Создать новый сценарий",
                        "Выбрать другой жанр",
                        "Изменить углублённость сюжета",
                    ]
                ),
            )
        elif answer == "":
            bot.send_message(
                user_id,
                "Сценарий создан!",
                reply_markup=create_keyboard(
                    [
                        "Создать новый сценарий",
                        "Выбрать другой жанр",
                        "Изменить углублённость сюжета",
                    ]
                ),
            )
        else:
            bot.send_message(
                user_id,
                answer,
                reply_markup=create_keyboard(
                    [
                        "Создать новый сценарий",
                        "Продолжить сценарий",
                        "Выбрать другой жанр",
                        "Изменить углублённость сюжета",
                    ]
                ),
            )
    else:  # Если токенов на продолжение не хватило
        bot.send_message(
            message.from_user.id,
            "Токенов на сценарий может не хватить:( Пожалуйста, попробуй укоротить свои желания. "
            "или задай новые",
            reply_markup=create_keyboard(["Создать новый сценарий"]),  # Предлагаем задать новый вопрос в рамках сессии
        )
        logging.info(
            f"Отправлено: {message.text}\nПолучено: Предупреждение о нехватке токенов"
        )


@bot.message_handler(commands=["debug"])
def send_logs(message):
    user_id = message.from_user.id
    if user_id in ADMINS:
        with open(LOGS_PATH, "rb") as f:
            bot.send_document(message.from_user.id, f)


bot.polling(non_stop=True)
