import re
import requests
from bs4 import BeautifulSoup
import json
import player
import redis
from selenium import webdriver
import csv
from datetime import date, datetime

class Parser:
    def __init__(self):
        self.seasons = [18]
        self.url = 'http://www.kicker.de/news/fussball/bundesliga/vereine/1-bundesliga/20XX-YY/vereine-liste.html'
        self.base = 'http://www.kicker.de'
        self.punkte = {"Note": {
                        "1,0" : 10,
                        "1,5" : 8,
                        "2,0" : 6,
                        "2,5" : 4,
                        "3,0" : 2,
                        "3,5" : 0,
                        "4,0" : -2,
                        "4,5" : -4,
                        "5,0" : -6,
                        "5,5" : -8,
                        "6,0" : -10,
        },
                    "Rot": -6,
                    "Gelb-Rot": -3,
                    "Tor": {
                        "Torwart": 6,
                        "Abwehr" : 5,
                        "Mittelfeld": 4,
                        "Sturm": 3},
                    "Assist": 1,
                    "Joker" : 1,
                    "Start": 2,
                    "zuNull": 2,
                    "SDS": 3}
    def parse(self, interactive=False):
        for season in self.seasons:
            print("parsing season", season)
            with open(str(season)+'.csv', 'w', newline='\n') as csvfile:
                writer = csv.writer(csvfile)
                season_data = self.parse_season(season, interactive)
                for data in season_data:
                    writer.writerow(data)
    def parse_season(self, season, interactive):
            r = requests.get(self.url.replace("XX", str(season)).replace("YY", str(season+1)))
            soup = BeautifulSoup(r.text)
            rows = soup.find("table", {"class":"tStat"}).find_all("tr")
            data = []
            for row in rows[1:]:
                if "tr_sep" not in row.get("class"):
                    club_name = row.find("a", {"class": "link verinsLinkBild"}).get_text()
                    club_squad_link = row.find_all("td")[2].find("a").get("href")
                    print("Parsing Squad", club_name, club_squad_link)
                    yield from self.parse_squad(club_squad_link, season, club_name, interactive)
    def parse_squad(self, url, season, club, interactive):
        r = requests.get(self.base + url)
        soup = BeautifulSoup(r.text)
        table = soup.find("table", {"summary":"Kader"}).find_all("tr")
        for row in table:
            if row.get("class") is not None and "tr_sep" not in row.get("class") and "zwischencaption" not in row.get("class"):
                player_link = row.find("a").get("href")
                yield from self.parse_player(player_link, season, club, interactive)

    def parse_player(self, url, season, club, interactive):
        r = requests.get(self.base + url)
        soup = BeautifulSoup(r.text)
        p_name = soup.find("h1").get_text()
        p_position = soup.find("tr", {"id":"ctl00_PlaceHolderContent_SpielerControl_SpielerControl_Spielerdaten_trPos"}).find_all("td")[1].get_text()
        p_age = soup.find("table", {"class":"infoBox"}).find(text = "Geboren am:").findNext("td").get_text()
        reference_date = datetime.strptime("01.08.20"+str(season), '%d.%m.%Y')
        age_date = datetime.strptime(p_age, '%d.%m.%Y')
        p_age = reference_date.year - age_date.year - ((reference_date.month, reference_date.day) < (age_date.month, age_date.day))
        p_club = club
        if interactive:
            yield p_name, p_position, p_age, p_club
        else:
            p_points = self.calc_points(soup, p_position)
            yield p_name, p_position, p_age, p_club, p_points
    def calc_points(self, soup, pos):
        # eingewechselt - spiele => start bonus
        # Gesamttore * Punkte für Tore
        # Gesamtassist => Punkte für Assists
        # eingewechselt => punkte für einwechselung
        # Noten einzeln parsen
        # gelb-rot / rot aus summary
        # zu Null bei TW
        table_summary = soup.find("tr", {"id": "a_sld_liga"}).find_all("td")
        pts_einwechsel = int(table_summary[7].get_text()) * self.punkte["Joker"]
        pts_tore = int(table_summary[3].get_text()) * self.punkte["Tor"][pos]
        pts_ass = int(table_summary[5].get_text()) * self.punkte["Assist"]
        pts_start = (int(table_summary[1].get_text().split("/")[0]) - pts_einwechsel) * self.punkte["Start"]
        pts_rot = int(table_summary[9].get_text()) * self.punkte["Rot"]
        pts_gelb_rot = int(table_summary[10].get_text()) * self.punkte["Gelb-Rot"]
        pts_note = 0
        pts_null = 0
        table_detail = soup.find("tr", {"id": "sld_liga_0"}).find("table").find_all("tr")
        for row in table_detail[1:]:
            if "tr_sep" not in row.get("class"):
                fields = row.find_all("td")
                pts_note += self.punkte["Note"].get(fields[4].get_text(), 0)
                if pos == "Torwart":
                    game_res = fields[3].find("a").get_text()
                    zu_null = int(game_res.split()[0].split(":")[1])
                    if zu_null == 0:
                        pts_null += self.punkte["zuNull"]
        return sum([pts_einwechsel, pts_tore, pts_ass, pts_start, pts_rot, pts_gelb_rot, pts_note, pts_null])
    def parse_interactive(self):
        print("Start Parsing Interactive")
        f = open("/home/marco/Downloads/kicker.html")
        soup = BeautifulSoup(f)
        f.close()
        table = soup.find("table", {"class":"tStat", "summary":""})
        rows = table.find_all("tr")
        for row in rows[1:]:
            if "tr_sep" not in row.get("class"):
                name = row.find("a", {"class":"link"})
                p_name = name.get_text()
                p_price = name.findNext("td").findNext("td").findNext("td").get_text().replace(",", ".")
                p_club = name.findNext("td").find("a", {"class":"link vrnMitLogo"}).get_text()
                yield p_name, p_price, p_club

def main():
    p = Parser()
    #p.parse_interactive()
    p.parse(interactive=True)

if __name__ == "__main__":
    main()
