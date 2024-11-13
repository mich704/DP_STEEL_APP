FROM python:3.11.6 AS python-base

WORKDIR /app

COPY DP_STEEL_PROJECT/requirements.txt /app/

RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY DP_STEEL_PROJECT /app/

FROM node:14 AS node-base

WORKDIR /frontend

COPY frontend/package*.json ./

RUN npm install

COPY frontend /frontend/

RUN npm run build

FROM python-base

COPY --from=node-base /frontend/build /app/frontend/build

EXPOSE 8000

ENV DJANGO_SETTINGS_MODULE=DP_STEEL_PROJECT.settings

CMD ["uvicorn", "DP_STEEL_PROJECT.asgi:application", "--host", "0.0.0.0", "--port", "8000"]