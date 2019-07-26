import re
import requests
from bs4 import BeautifulSoup
import json
import player
from selenium import webdriver
import csv
from datetime import date, datetime

class Parser:
    def __init__(self):
        self.seasons = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
        self.name_regex = re.compile("[\wöäüß\-]+", re.IGNORECASE)
        self.geb_regex = re.compile("[0-9]{2}\.[0-9]{2}\.[0-9]{4}")
        self.noten_regex = re.compile("[0-9],[0-9]")
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
                        "Tor": 6,
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
            with open(str(season).zfill(2)+'.csv', 'w', newline='\n') as csvfile:
                writer = csv.writer(csvfile)
                season_data = self.parse_season(season, interactive)
                for data in season_data:
                    writer.writerow(data)
    def parse_season(self, season, interactive):
            r = requests.get(self.url.replace("XX", str(season).zfill(2)).replace("YY", str(season+1).zfill(2)))
            soup = BeautifulSoup(r.text)
            rows = soup.find("table", {"class":"kick__table kick__table--ranking kick__table--alternate kick__table--resptabelle"}).find_all("tr")
            data = []
            for row in rows:
                club_name = row.find_all("td")[1].find("a").get_text()
                club_squad_link = row.find_all("td")[3].find("a").get("href")
                print("Parsing Squad", club_name, club_squad_link)
                yield from self.parse_squad(club_squad_link, season, club_name, interactive)
    def parse_squad(self, url, season, club, interactive):
        r = requests.get(self.base + url)
        soup = BeautifulSoup(r.text)
        table = soup.find("div", {"data-target":"squadContent"}).find_all("a")
        for row in table:
            if row.get("href") is not "#":
                player_link = row.get("href")
                #print("Parsing Player: ", player_link)
                yield from self.parse_player(player_link, season, club, interactive)

    def parse_player(self, url, season, club, interactive):
        r = requests.get(self.base + url)
        soup = BeautifulSoup(r.text)
        try:
            p_name = soup.find("h2").get_text()
            name = self.name_regex.findall(p_name)
            #print(name)
            p_name = ' '.join(name[::-1])
            p_position = soup.find(text="Position").findNext("td").get_text()
            p_age = soup.find(text = "Geboren am").findNext("td").get_text()
            p_age = self.geb_regex.search(p_age)[0]
            reference_date = datetime.strptime("01.08.20"+str(season).zfill(2), '%d.%m.%Y')
            age_date = datetime.strptime(p_age, '%d.%m.%Y')
            p_age = reference_date.year - age_date.year - ((reference_date.month, reference_date.day) < (age_date.month, age_date.day))
            p_club = club
            if interactive:
                yield p_name, p_position, p_age, p_club
            else:
                p_points = self.calc_points(soup, p_position, p_club)
                yield p_name, p_position, p_age, p_club, p_points
        except Exception as e:
            print("ERROR:", e)
            pass
    def calc_points(self, soup, pos, club):
        # eingewechselt - spiele => start bonus
        # Gesamttore * Punkte für Tore
        # Gesamtassist => Punkte für Assists
        # eingewechselt => punkte für einwechselung
        # Noten einzeln parsen
        # gelb-rot / rot aus summary
        # zu Null bei TW
        table_summary = soup.find("tr", {"class": "kick__js-open-saison-detail"}).find_all("td")
        pts_einwechsel = int(table_summary[7].get_text()) * self.punkte["Joker"]
        if pos == 'Unbekannt':
            pts_tore = int(table_summary[3].get_text())* self.punkte["Tor"]["Mittelfeld"]
        else:
            pts_tore = int(table_summary[3].get_text()) * self.punkte["Tor"][pos]
        pts_ass = int(table_summary[5].get_text()) * self.punkte["Assist"]
        pts_start = (int(table_summary[1].get_text().split("/")[0]) - pts_einwechsel) * self.punkte["Start"]
        pts_rot = int(table_summary[11].get_text()) * self.punkte["Rot"]
        pts_gelb_rot = int(table_summary[10].get_text()) * self.punkte["Gelb-Rot"]
        pts_note = 0
        pts_null = 0
        table_detail = soup.find_all("tr", {"class": "kick__vita__statistic--table-second--1 kick__vita__statistic--table-second"})
        for row in table_detail[1:]:
            fields = row.find_all("td")
            if len(fields) is 12:
                if self.noten_regex.search(fields[1].get_text()):
                    pts_note += self.punkte["Note"].get(self.noten_regex.search(fields[1].get_text())[0], 0)
                if pos == "Tor":
                    if row.find_all("div", {"class": "kick__v100-gameCell__team__name"})[0].get_text() is club:
                        zu_null = int(row.find_all("div", {"class": "kick__v100-scoreBoard__scoreHolder__score"})[1].get_text())
                    else:
                        zu_null = int(row.find_all("div", {"class": "kick__v100-scoreBoard__scoreHolder__score"})[0].get_text())
                    if zu_null == 0:
                        pts_null += self.punkte["zuNull"]
        #print([pts_einwechsel, pts_tore, pts_ass, pts_start, pts_rot, pts_gelb_rot, pts_note, pts_null])
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
    #p.parse(interactive=False)
    #res = p.parse_player("/bauer-robert-79879/spieler/1-bundesliga/2018-19/1-fc-nuernberg-81", 18, "1. FC Nürnberg", False)
    #res = p.parse_player("/timmy-simons-27977/spieler/1-bundesliga/2010-11/1-fc-nuernberg-81", 10, "1. FC Nürnberg", False)
    #[print(r) for r in res]

if __name__ == "__main__":
    main()
