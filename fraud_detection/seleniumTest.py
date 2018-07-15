from bs4 import BeautifulSoup
from selenium import webdriver
import time

browser = webdriver.Chrome()
input=open("dataset/unmatchType.txt")
for line in input:
    line = line.strip("\r\n")
    print(line)
    browser.get('https://db.yaozh.com/yibao?formname=%E5%90%84%E7%9C%81%E5%8C%BB%E4%BF%9D%E7%9B%AE%E5%BD%95&ybname='+line+'&jx1=&yplb1=&yblb1=%E5%85%A8%E9%83%A8&ybdq=%E5%85%A8%E9%83%A8&year=%E5%85%A8%E9%83%A8')
    page=browser.page_source
    cookies = browser.get_cookies()
    browser.add_cookie(cookies[0])  ###加入cookie操作
    time.sleep(10)
    input = browser.find_element_by_name('ybname')
    input.clear()
    input.send_keys(line)
    button=browser.find_element_by_xpath("//button[@type='submit']")
    soup = BeautifulSoup(page)
    pageSJ = soup.find_all('tr')
    f = open('SJ.txt', 'a')
    for i in pageSJ[1:]:
        f.write('\n')
        for item in i:
            if item not in ['\n', '\t', ' ']:
                # if item==None:#将空白项填入“None”
                #     f.write('None'+'|')
                # else:
                f.write(item.get_text(strip=True) + '\t')
    f.close()
browser.close()
input.close()