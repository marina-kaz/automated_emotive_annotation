# Веб-приложение для автоматического нанесения эмотивной разметки на тексты неформального интернет-дискурса

Разработано в рамках дипломного исследования. 

Для локального запуска необходимо выполнить следующие шаги: 

1. Установить требуемые зависимости

```
pip install -r requirements.txt
```

2. Запустить web-приложение

```
cd interpretable_nlp
python -m streamlit run steamlit-app
```

> Для корректной работы приложения, в папку `interpretable_nlp` необходимо поместить веса используемых моделей. 

Идея организации веб-приложения вдохновлена проектом [interpretable_nlp](https://github.com/CVxTz/interpretable_nlp)
