import json
import numpy as np
import re

state_abbrv_dict = None
state_dict = None
city_state_dict = None
countries = []

with open('data/location/state_abbrv_dict.json', 'r') as f:
    a = f.read()
state_abbrv_dict = json.loads(a)

with open('data/location/state_dict.json', 'r') as f:
    a = f.read()
state_dict = json.loads(a)

with open('data/location/city_state_dict.json', 'r') as f:
    a = f.read()
city_state_dict = json.loads(a)

with open('data/location/countries.txt') as f:
    for line in f:
        countries.append(line.rstrip())

with open('data/location/country_codes_dict.json', 'r') as f:
    a = f.read()
country_codes_dict = json.loads(a)


def getStates():
    return state_abbrv_dict

def findLocation(text):
    location = None
    
    if "washington d.c." in text.lower() or "washington dc" in text.lower() or "district of columbia" in text.lower():
        location = "DC"
        return location
    
    text = re.sub(r'[^\w\s]',' ', text).strip()
    text = " ".join(text.split())
    text_splitted = text.split(" ")
    
    if text_splitted[-1].upper() in state_abbrv_dict:
        location = text_splitted[-1].upper()
        #return location
    else:
        #for state in state_dict:
        for state in sorted(state_dict.keys()):
            if state in text.lower():
                location = state_dict[state]
                break
                #return location
        
        if location == None:
            for city in city_state_dict:
                if len(city.split(" ")) > 1:
                    if city in text.lower():
                        location = city_state_dict[city]
                        break
                        #return location
                else:
                    for word in text.lower().split(" "):
                        #word = re.sub(r'[^\w\s]','',word)
                        if word == city:
                            location = city_state_dict[city]
                            break
                            #return location
        
        country = None
        for c in countries:
            if c.lower() in text.lower():
                country = c
                break
                #return location
        
        if country == 'United States' or country == 'USA' or country == 'U.S.A' or country == 'U.S.A.':
            country = 'United States'
        elif country == 'England' or country == 'Scotland' or country == 'Wales' or country == 'UK' or country == 'U.K.' or country == 'United Kingdom':
            country = 'Great Britain'
        
        '''
        if (location != None and (country != 'United States' and country != 'USA' and country != 'U.S.A' and country != None)):
            location = country
        elif location == None:
            location = country
        '''
        if location == None:
            location = country
    
    return location


def findLocationFromTweet(tweet_geoname, tweet_cc, est_user_loc, user_loc):
    location = None
    if tweet_geoname != None and tweet_geoname != 'N':
        # take state or country from tweet_geoname
        if tweet_cc == 'US':
            location = findLocation(tweet_geoname)
            if location == None:
                location = 'United States'
        else:
            if tweet_cc in country_codes_dict:
                location = country_codes_dict[tweet_cc]
        
        #print('tweet_cc: ', tweet_cc, 'tweet_geoname: ', tweet_geoname, ' --> ', location)
        #print('({}, {})'.format(tweet_geoname, tweet_cc), '-->', location, ' ---- ', '({})'.format(user_loc), '-->', est_user_loc)
        
    return location


