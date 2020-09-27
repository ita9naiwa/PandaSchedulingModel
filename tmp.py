import requests
from bs4 import BeautifulSoup as bs

url = "https://finance.naver.com/sise/sise_market_sum.nhn"
response = requests.get(url)
html = bs(response.content, 'html.parser')


table = html.find("table", {"class":"type_2"})
tbody = table.find("tbody")
trs = tbody.findAll("tr")

def fit(s, t=10):
    return s + " " * ((t - len(s)))
for tr in trs:
    tds = tr.findAll("td")
    elems = []
    if len(tds) > 5:
        elem = [tds[0].text, tds[1].text, tds[6].text, tds[2].text]
        elems.append(elem)
        f = [fit(x) for x in elem]
        print("%s%s%s%s" % (f[0], f[1], f[2], f[3]))