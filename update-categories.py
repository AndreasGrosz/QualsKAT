import os
import sys
import configparser
from data_processing import extract_categories, save_categories_to_csv


def update_categories(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    categories = extract_categories(config)
    save_categories_to_csv(categories, config)

if __name__ == "__main__":
    config_path = 'config.txt'  # Pfad zur Konfigurationsdatei
    update_categories(config_path)
