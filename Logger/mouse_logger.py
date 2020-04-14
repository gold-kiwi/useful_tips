import os
import sys
from PIL import Image
import time
import datetime
#import mouse
from pynput.mouse import Listener
import pyautogui as pgui


###
# It logs mouse position and makes screenshot every click, double click and drag/drop event.
# It useful for making efficient manual to instruct people some PC work from the viw point of visualizing.
# A mouse cursor image is needed.
###
class Track_Mouse():
    def __init__(self,save_dir='',double_click_interval=0.5,path_to_cursor_img='cursor_img.png'):
        self.flag = True
        self.shot_counter = 0
        self.x_down = 0
        self.y_down = 0
        self.save_dir = save_dir
        self.saved_file_list = []
        self.log = []
        self.cursor = Image.open(path_to_cursor_img)
        self.double_click_interval = double_click_interval
        self.click_time = datetime.datetime.now()
        self.pre_click_time = datetime.datetime.now()
        self.dt = 1
        self.s_tmp = pgui.screenshot()
        os.makedirs(save_dir,exist_ok=True)

    def get_flag(self):
        return self.flag

    def switch_flag(self):
        self.flag = not self.flag

    def save_screenshot(self,x,y,time_stamp,status,use_another_shot=False,another_shot=-1):
        if use_another_shot:
            s = another_shot
        else:
            s = pgui.screenshot()

        a = Image.new('RGBA',s.size, (255,255,255,0))
        a.paste(self.cursor, (x,y))
        s.paste(a,(0,0),mask=a)

        time_str = "{0:%Y_%m%d_%H_%M_%S_%f}".format(time_stamp)[:-2]
        final_name = 'screenshot_%3d_%s_%s_%d_%d.png' % (self.shot_counter,time_str,status,x,y)
        self.saved_file_list.append(final_name)
        s.save(self.save_dir + final_name)
        self.shot_counter += 1

    def save_log(self):
        with open(self.save_dir + 'log.txt', mode="w") as f:
            for line in self.log:
                f.writelines(str(line) + '\n')

    def on_move(self,x,y):
        pass

    def on_click(self,x,y,button,pressed):
        if x >= 0 and y >= 0 and (x+y) < 10:
            print('RECORD FINISHED')
            self.save_log()
            self.flag = False
            return False

        else:
            print(x,y)
            if pressed:
                self.x_down = x
                self.y_down = y

                new_click = datetime.datetime.now()
                self.dt = new_click - self.click_time
                self.pre_click_time = self.click_time
                self.click_time = new_click
                self.s_tmp = pgui.screenshot()
                return False

            else:
                if x == self.x_down and y == self.y_down:
                    if self.dt.total_second() < self.double_click_interval:
                        self.log[-1] = ['double_click',x,y]
                        os.remove(self.save_dir + self.save_file_list[-1])
                        self.save_file_list.pop(-1)
                        self.shot_counter -= 1
                        self.save_screenshot(x,y,self.click_time,'double_click')

                    else:
                        self.log.append(['click',x,y])
                        self.save_screenshot(x,y,self.click_time,'click')

                else:
                    self.log.append(['drag',self.x_down,self.y_down])
                    self.save_screenshot(self.x_dwon,self.y_dwon,self.pre_click_time,'drag',use_another_shot=True,another_shot=self.s_tmp)
                    self.log.append(['drop',x,y])
                    self.save_screenshot(x,y,self.click_time,'drop')

    def on_scroll(self,x,y,dx,dy):
        pass

def main():
    dt_now = datetime.datetime.now()
    time_str = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

    shot_save_dir = '.\\mouse_record\\' + time_str + '\\'
    print('RECORD START')
    print('SAVE AT : ', shot_save_dir)
    tm = Track_Mouse(save_dir=shot_save_dir)

    while tm.get_flag():
        with Listener(on_move=tm.on_move,on_click=tm.on_click,on_scroll=tm.on_scroll) as listener:
            listener.join()

if __name__ == '__main__':
    main()
