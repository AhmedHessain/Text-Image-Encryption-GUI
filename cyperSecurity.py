import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QComboBox, QLineEdit, QPushButton, QFileDialog, QStackedWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic
import numpy as np

class MyForm(QMainWindow):
    def __init__(self):
        super(MyForm, self).__init__()
        uic.loadUi("Cyber Security.ui", self)

        # Accessing widgets defined in the UI file
        self.plain_text_edit = self.findChild(QTextEdit, 'plainText')
        self.cipher_list = self.findChild(QComboBox, 'cipher_list')
        self.key_text = self.findChild(QLineEdit, 'keyText')
        self.encrypt_button = self.findChild(QPushButton, 'encryptButton')
        self.decrypt_button = self.findChild(QPushButton, 'decryptButton')
        self.encrypted_text_edit = self.findChild(QTextEdit, 'encryptedText')
        self.load_image_btn = self.findChild(QPushButton, 'loadImgBtn_4')
        self.encrypt_btn = self.findChild(QPushButton, 'encryptImgBtn_4')
        self.decrypt_btn = self.findChild(QPushButton, 'decryptImgBtn_4')
        self.load_encrypted_btn = self.findChild(QPushButton, 'loadEcryptedImgBtn_5')
        self.save_encrypted_btn = self.findChild(QPushButton, 'loadEcryptedImgBtn_6')
        self.save_decrypted_btn = self.findChild(QPushButton, 'loadEcryptedImgBtn_7')
        self.selectTextPageBtn = self.findChild(QPushButton,'selectTextPageBtn')
        self.selectImgPageBtn = self.findChild(QPushButton,'selectImgPageBtn')
        self.stackedWidget = self.findChild(QStackedWidget, 'stackedWidget')
        self.key_Image = self.findChild(QLineEdit, 'keyImage')
        # Connect signals to slots
        self.encrypt_button.clicked.connect(self.encrypt)
        self.decrypt_button.clicked.connect(self.decrypt)
        self.cipher_list.currentIndexChanged.connect(self.update_key_value)
        self.load_image_btn.clicked.connect(self.load_image)
        self.encrypt_btn.clicked.connect(self.encrypt_image)
        self.decrypt_btn.clicked.connect(self.decrypt_image)
        self.load_encrypted_btn.clicked.connect(self.load_encrypted_image)
        self.save_encrypted_btn.clicked.connect(self.save_encrypted_image)
        self.save_decrypted_btn.clicked.connect(self.save_decrypted_image)

        self.selectTextPageBtn.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.TextEncryptionPage))
        self.selectImgPageBtn.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.ImageEncryptionPage))

        self.set_image(self.decryptedimg, "senua.jpg")
        self.set_image(self.encryptedImg, "encryptedSenua.png")

    def update_key_value(self):
        cipher_type = self.cipher_list.currentText()
        if cipher_type == 'Caesar':
            self.key_text.setText('15')
        elif cipher_type == 'Monoalphabetic':
            self.key_text.setText('noatrbecfuxdqgylkhvijmpzsw')
        elif cipher_type == 'Polyalphabetic':
            self.key_text.setText('12')
        elif cipher_type == 'Playfair':
            self.key_text.setText('monarchy')
            
    def encrypt(self):
        plain_text = self.plain_text_edit.toPlainText()
        cipher_type = self.cipher_list.currentText()
        key = self.key_text.text()

        if cipher_type == 'Caesar':
            encrypted_text = self.caesar_encrypt(plain_text, int(key))
        elif cipher_type == 'Monoalphabetic':
            encrypted_text = self.monoalphabetic_encrypt(plain_text, key)
        elif cipher_type == 'Polyalphabetic':
            encrypted_text = self.polyalphabetic_encrypt(plain_text, int(key))
        elif cipher_type == 'Playfair':
            encrypted_text = self.playfair_encrypt(plain_text, key)
        else:
            encrypted_text = 'Invalid cipher type'

        self.encrypted_text_edit.setPlainText(encrypted_text)

    def decrypt(self):
        encrypted_text = self.encrypted_text_edit.toPlainText()
        cipher_type = self.cipher_list.currentText()
        key = self.key_text.text()

        if cipher_type == 'Caesar':
            decrypted_text = self.caesar_decrypt(encrypted_text, int(key))
        elif cipher_type == 'Monoalphabetic':
            decrypted_text = self.monoalphabetic_decrypt(encrypted_text, key)
        elif cipher_type == 'Polyalphabetic':
            decrypted_text = self.polyalphabetic_decrypt(encrypted_text, int(key))
        elif cipher_type == 'Playfair':
            decrypted_text = self.playfair_decrypt(encrypted_text, key)
        else:
            decrypted_text = 'Invalid cipher type'

        self.plain_text_edit.setPlainText(decrypted_text)


    def caesar_encrypt(self, text, key):
        result = ''
        text = text.lower()
        for char in text:
            if char.isalpha():
                encrypted_char = chr((ord(char) - 97 + key) % 26 + 97)
                result += encrypted_char
            else:
                result += char
        return result

    def caesar_decrypt(self, text, key):
        result = ''
        text = text.lower()
        for char in text:
            if char.isalpha():
                decrypted_char = chr((ord(char) - 97 - key) % 26 + 97)
                result += decrypted_char
            else:
                result += char
        return result

    def monoalphabetic_encrypt(self, text, key):
        text = text.lower()
        result = ''
        key_mapping = dict(zip('abcdefghijklmnopqrstuvwxyz', key.lower()))
        for char in text:
            if char.isalpha():
                result += key_mapping[char]
            else:
                result += char
        return result

    def monoalphabetic_decrypt(self, text, key):
        text = text.lower()
        result = ''
        key_mapping = dict(zip(key.lower(),'abcdefghijklmnopqrstuvwxyz'))
        for char in text:
            if char.isalpha():
                result += key_mapping[char]
            else:
                result += char
        return result

    def polyalphabetic_encrypt(self, text, key):
        text = text.lower()
        result = ''
        for char in text:
            if char.isalpha():
                encrypted_char = chr((ord(char) - 97 + key) % 26 + 97)
                key = ord(char) - 97
                result += encrypted_char
            else:
                result += char
        return result

    def polyalphabetic_decrypt(self, text, key):
        text = text.lower()
        result = ''
        for char in text:
            if char.isalpha():
                decrypted_char = chr((ord(char) - 97 - key) % 26 + 97)
                key = ord(decrypted_char) - 97
                result += decrypted_char
            else:
                result += char
        return result

    def generate_playfair_matrix(self,key):
        key = key.lower().replace("j", "i")  # Convert to lowercase and replace j with i
        alphabet = "abcdefghiklmnopqrstuvwxyz"
        
        # Create the Playfair matrix
        matrix = [[0] * 5 for _ in range(5)]
        key_chars = [ch for ch in key] + [ch for ch in alphabet if ch not in set(key)]
        
        for i in range(5):
            for j in range(5):
                matrix[i][j] = key_chars[i * 5 + j]
        
        return matrix

    def find_position(self,matrix, char):
        for i in range(5):
            for j in range(5):
                if matrix[i][j] == char:
                    return i, j

    def playfair_encrypt(self,plain_text, key):
        matrix = self.generate_playfair_matrix(key)
        encrypted_text = ""
        plain_text = plain_text.lower().replace("j", "i")
        plain_text_list = " ".join(plain_text.split()).split(" ")

        for text in plain_text_list:
            # Prepare the plain text by inserting an 'x' between repeated letters and padding if necessary
            pairs = [text[i:i+2] for i in range(0, len(text), 2)]
            for i in range(len(pairs)):
                if len(pairs[i]) == 1:
                    pairs[i] += "z"
                elif pairs[i][0] == pairs[i][1]:
                    pairs[i] = pairs[i][0] + "x" + pairs[i][1]

            # Encrypt each pair
            for pair in pairs:
                row1, col1 = self.find_position(matrix, pair[0])
                row2, col2 = self.find_position(matrix, pair[1])

                # Same row, shift columns to the right
                if row1 == row2:
                    encrypted_text += matrix[row1][(col1 + 1) % 5] + matrix[row2][(col2 + 1) % 5]
                # Same column, shift rows down
                elif col1 == col2:
                    encrypted_text += matrix[(row1 + 1) % 5][col1] + matrix[(row2 + 1) % 5][col2]
                # Form a rectangle, swap columns
                else:
                    encrypted_text += matrix[row1][col2] + matrix[row2][col1]
            encrypted_text += " "
        return encrypted_text

    def playfair_decrypt(self,encrypted_text, key):
        matrix = self.generate_playfair_matrix(key)
        decrypted_text = ""
        encrypted_text = encrypted_text.lower().replace("j", "i")
        encrypted_text_list = " ".join(encrypted_text.split()).split(" ")

        for text in encrypted_text_list:
        # Decrypt each pair
            pairs = [text[i:i+2] for i in range(0, len(text), 2)]
            for pair in pairs:
                row1, col1 = self.find_position(matrix, pair[0])
                row2, col2 = self.find_position(matrix, pair[1])

                # Same row, shift columns to the left
                if row1 == row2:
                    decrypted_text += matrix[row1][(col1 - 1) % 5] + matrix[row2][(col2 - 1) % 5]
                # Same column, shift rows up
                elif col1 == col2:
                    decrypted_text += matrix[(row1 - 1) % 5][col1] + matrix[(row2 - 1) % 5][col2]
                # Form a rectangle, swap columns
                else:
                    decrypted_text += matrix[row1][col2] + matrix[row2][col1]
            decrypted_text += " "
        return decrypted_text

    def set_image(self, label, path):
        pixmap = QPixmap(path)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        
    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.set_image(self.decryptedimg, file_path)

    def load_encrypted_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Encrypted Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.set_image(self.encryptedImg, file_path)


    def encrypt_decrypt_process(self,image_array):
        # Convert the image to a NumPy array with dtype np.uint8
        if image_array.shape[2] == 4:
            # Extract RGB channels (exclude the alpha channel)
            image_array = image_array[:, :, :3]
        
        image_array = image_array[:,:,[2,1,0]]
        seed = int(self.key_Image.text())
        # Initialize LCG with a seed derived from the key
        lcg = LCG(seed=seed)

        # Flatten the array for simplicity
        flat_array = image_array.flatten()
        # Generate a key stream using the LCG
        key_stream = np.array([lcg.next() % 256 for _ in range(len(flat_array))], dtype=np.uint8)
        # Perform XOR operation on flattened array
        processed_array = np.bitwise_xor(flat_array, key_stream)
        # Reshape the array back to its original shape
        processed_img_array = processed_array.reshape(image_array.shape)
        return processed_img_array

    def encrypt_image(self):
        encrypted_pixmap = self.decryptedimg.pixmap()
        encrypted_image_array = self.pixmap_to_array(encrypted_pixmap)
        encrypted_result = self.encrypt_decrypt_process(encrypted_image_array)
        self.encryptedImg.setPixmap(self.array_to_image(encrypted_result))


    def pixmap_to_array(self, pixmap):
            # Convert QPixmap to NumPy array
            image = pixmap.toImage()
            byte_format = image.format()
            if byte_format == QImage.Format_RGB32:
                size = image.byteCount()
                data = image.bits().asstring(size)
                return np.frombuffer(data, dtype=np.uint8).reshape(image.height(), image.width(), 4)
            elif byte_format == QImage.Format_RGB888:
                size = 3 * image.width() * image.height()
                data = image.bits().asstring(size)
                return np.frombuffer(data, dtype=np.uint8).reshape(image.height(), image.width(), 3)
            else:
                raise ValueError(f"Unsupported image format: {byte_format}")


    def array_to_image(self,array):
        if array.ndim == 3:
            height, width, channels = array.shape
        elif array.ndim == 2:
            height, width = array.shape
            channels = 1
            array = array.reshape((height, width, 1))
        else:
            raise ValueError("Unsupported array shape")

        if channels == 1:
            q_image = QImage(array.data, width, height, QImage.Format_Grayscale8)
        elif channels == 3:
            q_image = QImage(array.data, width, height, channels * width, QImage.Format_RGB888)
        else:
            raise ValueError("Unsupported number of channels")

        pixmap = QPixmap.fromImage(q_image)
        return pixmap

    
    def decrypt_image(self):
        encrypted_pixmap = self.encryptedImg.pixmap()
        encrypted_image_array = self.pixmap_to_array(encrypted_pixmap)
        encrypted_result = self.encrypt_decrypt_process(encrypted_image_array)
        self.decryptedimg.setPixmap(self.array_to_image(encrypted_result))

    def save_encrypted_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save Encrypted Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            pixmap = self.encryptedImg.pixmap()
            pixmap.save(file_path)

    def save_decrypted_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save Decrypted Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            pixmap = self.decryptedimg.pixmap()
            pixmap.save(file_path)

class LCG:
    def __init__(self, seed, a=1664525, c=1013904223, m=2**32):
        self.state = seed
        self.a = a
        self.c = c
        self.m = m

    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyForm()
    window.show()
    sys.exit(app.exec_())
