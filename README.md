# Cryptography App

## Overview

This graphical application provides a user-friendly interface for encrypting and decrypting both text and images using various cryptographic algorithms. The application is built using PyQt5, offering a seamless experience with buttons, text fields, and image display areas.

## Text Encryption and Decryption

### Caesar Cipher
- Shifts each letter in the text by a fixed number of positions down the alphabet.
- Supports both encryption and decryption.

### Monoalphabetic Cipher
- Substitutes each letter in the text with another letter based on a user-provided mapping.
- Allows users to encrypt and decrypt using a monoalphabetic substitution cipher.

### Polyalphabetic Cipher
- Similar to Caesar cipher but uses multiple substitution alphabets based on a given key.
- Supports both encryption and decryption.

### Playfair Cipher
- Encrypts pairs of letters at a time, enhancing security.
- Users can perform encryption and decryption operations.

## Image Encryption and Decryption

### Linear Congruential Generator (LCG)
- Generates a pseudo-random sequence of numbers.
- Used to create a key stream for XOR-based encryption and decryption of images.

### Image Encryption and Decryption
- Functionality to load, encrypt, and display images.
- Users can load encrypted images and decrypt them for viewing.

### Loading and Saving Images
- Allows users to load images from their computer.
- Supports saving both encrypted and decrypted images.

## GUI Components

- Buttons, input fields, and areas for displaying original, encrypted, and decrypted images.
- Dropdown menu for selecting different encryption algorithms.
- Stacked widget for switching between text and image encryption pages.

## UI Navigation

- Buttons (`selectTextPageBtn` and `selectImgPageBtn`) for easy navigation between text and image encryption pages.

## Initialization

- Initializes GUI components and connects them to corresponding functions.
- Sets up initial images in the interface.

## Execution

- The script starts the PyQt application, displaying the graphical interface, and runs an event loop to handle user interactions.

## Usage

1. Clone the repository.
2. Install dependencies.
3. Run the application script.

## License

This project is open-source and uses the MIT License, allowing others to use, modify, and distribute the code.

Feel free to explore and experiment with different encryption algorithms using this application!
