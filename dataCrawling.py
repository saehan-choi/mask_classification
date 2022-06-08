# 동작을 원할시 conda activate crawling 하세요
# 시작안했습니다. -> 구글에도 이미지 데이터가 몇개없네요 .. ㅠㅠ

from lib2to3.pgen2 import driver
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
import urllib.request
import time

import random

# example of insert your address : CRAWLING/crawling_dataset2
# изображение без маски -> 이게 좀 많았음

num = random.random()
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Chrome(options=options)

keyword = str(input("insert keyword for searching : "))#get keyword to search

driver.get("https://www.google.co.kr/imghp?hl=ko")##open google image search page
driver.maximize_window()##웹브라우저 창 화면 최대화
time.sleep(1)
driver.find_element_by_css_selector("input.gLFyf").send_keys(keyword) #send keyword
driver.find_element_by_css_selector("input.gLFyf").send_keys(Keys.RETURN)##send Keys.RETURN


last_height = driver.execute_script("return document.body.scrollHeight") #initialize standard of height first
while True: #break가 일어날 때 까지 계속 반복
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") #페이지 스크롤 시키기

    time.sleep(1.5)

    new_height = driver.execute_script("return document.body.scrollHeight") ## update new_height
    if new_height == last_height:#이전 스크롤 길이와 현재의 스크롤 길이를 비교
        try:
            driver.find_element_by_css_selector(".mye4qd").click() ## click more button 더보기 버튼이 있을 경우 클릭
        except:
            break # 더보기 버튼이 없을 경우는 더 이상 나올 정보가 없다는 의미이므로 반복문을 break
    last_height = new_height ##last_height update

i=0
list = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")##thumnails list
print(len(list)) #print number of thumnails

address = str(input("insert your address : "))# 파일을 저장할 주소를 입력받기
for img in list:
    i += 1
    try:
        imgurl = img.get_attribute("src") # get thumnails address list
        # time.sleep(1)

        urllib.request.urlretrieve(imgurl,address+"/"+str(keyword)+str(i)+".jpg") # download images in address folder

    except: #저장이 불가능할경우 그냥  pass
        pass

