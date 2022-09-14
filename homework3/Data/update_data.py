import requests

url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"

r = requests.get(url)

with open("dpc-covid19-ita-regioni.csv", "wb") as f:
	f.write(r.content)
