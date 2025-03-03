-- Création de la base de données
CREATE DATABASE textes_db;

-- Connexion à la base de données (à exécuter après la création)
\c textes_db;

-- Création de l'utilisateur avec mot de passe
CREATE USER user WITH PASSWORD 'mdp';

-- Donner les privilèges nécessaires à l'utilisateur
GRANT ALL PRIVILEGES ON DATABASE textes_db TO user;

-- Création de la table pour stocker les fichiers JSON
CREATE TABLE textes (
    id SERIAL PRIMARY KEY,
    texte JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Donner les droits sur la table
GRANT ALL PRIVILEGES ON TABLE textes TO user;
