# A Hello World NodeJs szerverhez tartozó Dockerfile

# Felhasználja a már meglévő node image-t
FROM node:14 
WORKDIR /usr/src/app
# Függőségek és app információk másolása, majd telepítése
COPY package*.json ./
RUN npm install 
COPY . .
# A Docker konténer összeköttetése a külvilággal
EXPOSE 8080
# A szerver elindítása
CMD ["node", "app.js"]
