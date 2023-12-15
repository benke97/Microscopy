#%%
import pyautogui
import time


class DrProbeRunner():
    def __init__(self):
        self.dataset=6
        self.confidence_level = 0.8

    def get_file(self,i):
        if i < 10:
            return "data_000"+str(i)+".cif"
        elif i < 100:
            return "data_00"+str(i)+".cif"
        elif i < 1000:
            return "data_0"+str(i)+".cif"
        else:
            return "data_"+str(i)+".cif"

    def start_drprobe(self):
        #start drprobe
        pyautogui.press('win')
        pyautogui.write('dr')
        pyautogui.press('enter')

        # Wait for the application to open
        time.sleep(10)

    def sample_setup(self,file):
        # Attempt to locate the button
        try:
            button_location = pyautogui.locateOnScreen('sample_setup.png', confidence=self.confidence_level)
            if button_location is not None:
                button_point = pyautogui.center(button_location)
                pyautogui.click(button_point)
            else:
                print("Sample setup button could not be found.")
        except pyautogui.ImageNotFoundException:
            print("Sample setup button could not be found.")
        pyautogui.moveTo(300,300)
        time.sleep(2)

        try:
            button_location = pyautogui.locateOnScreen('select_file.png', confidence=self.confidence_level)
            if button_location is not None:
                button_point = pyautogui.center(button_location)
                pyautogui.click(button_point)
            else:
                print("Select file button could not be found.")
        except pyautogui.ImageNotFoundException:
            print("Select file button could not be found.")

            #type data_0001.cif + enter
        time.sleep(2)
        pyautogui.write(file)
        pyautogui.press('enter')
        time.sleep(2)
        try:
            button_location = pyautogui.locateOnScreen('horizontally.png', confidence=self.confidence_level)
            if button_location is not None:
                button_point = pyautogui.center(button_location)
                pyautogui.click(button_point)
            else:
                print("Horizontally box could not be found.")
        except pyautogui.ImageNotFoundException:
            print("Horizontally box could not be found.")
        time.sleep(2)
        pyautogui.press('right')
        pyautogui.press('right')
        pyautogui.press('right')
        pyautogui.press('right')
        pyautogui.press('backspace')
        pyautogui.press('backspace')
        pyautogui.press('backspace')
        pyautogui.press('backspace')
        pyautogui.write('1200')

        time.sleep(2)
        try:
            button_location = pyautogui.locateOnScreen('vertically.png', confidence=self.confidence_level)
            if button_location is not None:
                button_point = pyautogui.center(button_location)
                pyautogui.click(button_point)
            else:
                print("Vertically box could not be found.")
        except pyautogui.ImageNotFoundException:
            print("Vertically box could not be found.")
        time.sleep(2)
        pyautogui.press('right')
        pyautogui.press('right')
        pyautogui.press('right')
        pyautogui.press('right')
        pyautogui.press('backspace')
        pyautogui.press('backspace')
        pyautogui.press('backspace')
        pyautogui.press('backspace')
        pyautogui.write('1200')
        time.sleep(2)
        try:
            button_location = pyautogui.locateOnScreen('variants_per_slice_1.png', confidence=self.confidence_level)
            if button_location is not None:
                button_point = pyautogui.center(button_location)
                pyautogui.click(button_point)
            else:
                button_location = pyautogui.locateOnScreen('variants_per_slice_10.png', confidence=self.confidence_level)
                if button_location is not None:
                    button_point = pyautogui.center(button_location)
                    pyautogui.click(button_point)
                else:
                    print("Variants per slice box could not be found.")
        except pyautogui.ImageNotFoundException:
            print("Variants per slice box could not be found.")
        pyautogui.press('right')
        pyautogui.press('right')
        pyautogui.press('backspace')
        pyautogui.press('backspace')
        pyautogui.write('10')
        time.sleep(2)
        try:
            button_location = pyautogui.locateOnScreen('start_slice_creation.png', confidence=self.confidence_level)
            if button_location is not None:
                button_point = pyautogui.center(button_location)
                pyautogui.click(button_point)
            else:
                print("Start slice creation button could not be found.")
        except pyautogui.ImageNotFoundException:
            print("Start slice creation button could not be found.")
            time.sleep(2)

        while(1):
            time.sleep(2)
            try:
                button_location = pyautogui.locateOnScreen('slice_calculation_finished.png', confidence=self.confidence_level)
                if button_location is not None:
                    button_point = pyautogui.center(button_location)
                    print("slice calculation has finished")
                    break
                else:
                    print("slice calculation has not finished")

            except pyautogui.ImageNotFoundException:
                print("slice calculation has not finished")
            
        time.sleep(2)
        try:
            button_location = pyautogui.locateOnScreen('ok_slice_creation.png', confidence=self.confidence_level)
            if button_location is not None:
                button_point = pyautogui.center(button_location)
                pyautogui.click(button_point)
            else:
                print("ok after slice creation finished not found")
        except pyautogui.ImageNotFoundException:
            print("ok after slice creation finished not found")

        time.sleep(2)
        try:
            button_location = pyautogui.locateOnScreen('OK.png', confidence=self.confidence_level)
            if button_location is not None:
                button_point = pyautogui.center(button_location)
                pyautogui.click(button_point)
            else:
                print("OK not found")
        except pyautogui.ImageNotFoundException:
            print("OK not found")
        
        print("sample setup finished")
        time.sleep(2)

    def calculation_setup(self):
        print("calculation setup started")

    def run_calculation(self,i):
        try:
            button_location = pyautogui.locateOnScreen('start_calculation.png', confidence=self.confidence_level)
            if button_location is not None:
                button_point = pyautogui.center(button_location)
                pyautogui.click(button_point)
            else:
                print("start calculation button could not be found.")
        except pyautogui.ImageNotFoundException:
            print("start calculation button could not be found.")
        pyautogui.moveTo(300,300)
        time.sleep(5)
        if i > 0:
            try:
                button_location = pyautogui.locateOnScreen('yes.png', confidence=self.confidence_level)
                if button_location is not None:
                    button_point = pyautogui.center(button_location)
                    print("SUCCESS")
                    pyautogui.click(button_point)
                else:
                    print("yes button could not be found.")
            except pyautogui.ImageNotFoundException:
                print("yes button could not be found.")

        pyautogui.moveTo(300,300)
        time.sleep(3)
        try:
            button_location = pyautogui.locateOnScreen('yes.png', confidence=self.confidence_level)
            if button_location is not None:
                button_point = pyautogui.center(button_location)
                pyautogui.click(button_point)
            else:
                print("yes button could not be found.")
        except pyautogui.ImageNotFoundException:
            print("yes button could not be found.")
        
        pyautogui.moveTo(300,300)
        while(1):
            time.sleep(2)
            try:
                button_location = pyautogui.locateOnScreen('calculation_finished.png', confidence=self.confidence_level)
                if button_location is not None:
                    button_point = pyautogui.center(button_location)
                    print("calculation has finished")
                    break
                else:
                    print("calculation has not finished")

            except pyautogui.ImageNotFoundException:
                print("calculation has not finished")

    def save_data(self):
        print("save data started")

    def generate_simulated_images(self):
        #start drprobe
        self.start_drprobe()
        for i in range(self.dataset):
            file = self.get_file(i)
            self.sample_setup(file)
            self.calculation_setup()
            self.run_calculation(i)
            self.save_data()
#%%
if __name__ == "__main__":
    drprobe = DrProbeRunner()
    drprobe.generate_simulated_images()
# %%
