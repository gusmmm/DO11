#!/usr/bin/env python3
"""
Main entry point for the Data Science Workflow application.
"""
from utils.switch import MenuController

def main():
    """Main function to start the application."""
    menu = MenuController()
    menu.show_main_menu()

if __name__ == "__main__":
    main()
