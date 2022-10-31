from flask import Flask, request, render_template, request, session
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import pandas as pd
import dill
import joblib
import re 
from tensorflow.keras.models import load_model

import spacy
import nltk, re
from nltk.corpus import wordnet 
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class EntityMatcher(object):
	name = 'entity_matcher'

	def __init__(self, nlp, terms, label):
		patterns = [nlp.make_doc(text) for text in terms]
		self.matcher = PhraseMatcher(nlp.vocab)
		self.matcher.add(label, None, *patterns)

	def __call__(self, doc):
		matches = self.matcher(doc)
		for match_id, start, end in matches:
			span = Span(doc, start, end, label=match_id)
			doc.ents = list(doc.ents) + [span]
		return doc
def getHypernyms(token):
	hypernyms = []
	synsets = wordnet.synsets(token)
	for synset in synsets:
		for h in synset.hypernyms():
			for l in h.lemmas():
				hypernyms.append(l.name())
	return list(set(hypernyms))
def getHyponyms(token):
	hyponyms = []
	synsets = wordnet.synsets(token)
	for synset in synsets:
		for h in synset.hyponyms():
			for l in h.lemmas():
				hyponyms.append(l.name())
	return list(set(hyponyms))
def getMeronyms(token):
	meronyms = []
	synsets = wordnet.synsets(token)
	for synset in synsets:
		for h in synset.part_meronyms():
			for l in h.lemmas():
				meronyms.append(l.name())
	return list(set(meronyms))
def getHolonyms(token):
	holonyms = []
	synsets = wordnet.synsets(token)
	for synset in synsets:
		for h in synset.member_holonyms():
			for l in h.lemmas():
				holonyms.append(l.name())
	return list(set(holonyms))

if __name__ == '__main__':
	synonyms = {
	'killing' : {'shot','fired','killed','murdered', 'murder', 'killing', 'vote_out', 'cleanup', 'defeat', 'vote_down', 'pour_down', 'drink_down', 'kill', 'obliterate', 'wipe_out', 'sidesplitting', 'pop', 'down', 'toss_off', 'belt_down', 'shoot_down', 'bolt_down', 'violent_death', 'stamp_out', 'putting_to_death'},
	'kidnap' : {'kidnappers', 'kidnapping', 'forcefully','kidnap', 'abduct', 'capture','seize', 'snatch', 'kidnapped', 'abducted', 'captured', 'snatched','hostage', 'ransom', 'taken', 'missing', 'disappearance', 'vanished', 'disappeared'},
	'acquisition' : {'bought','buy','purchased','acquired','acquire','purchase','takeover','merger','selling', 'sold', 'sell'},
	'award' : {"award", "awarded", "Award", "Awarded", "wins", "honored"},
	'scandals' :{"harassment", "affair", "sexual", "rumour", "rumoured", "accused", "accuse", "accuses"},
	'disaster' : {'flood','floods','earthquake','earthquakes','hurricane','hurricanes','storms','storm','landslide','landslides','avalanche','avalanches','tsunami','blizzard','tornado','tornadoes','drought','droughts','thunderstorm','thunderstorms'},
	'injury' : {'injury','injured','injuries', 'ruled', 'out'},
	'diseases':{'outbreak','disease','widespread', 'epidemic', 'pandemic', 'spread', 'deaths'}
	}
	nlp = spacy.load('en_core_web_sm')


	filename = 'corpus.txt'
	#filename = "The bullet responsible for killing Ron Helus from Ventura County during November's mass shooting at the Borderline Bar & Grill was fired by Ian David Long, authorities said Friday."
	file = open(filename, 'r', encoding="utf-8")
	document = file.read()
	articles = document.split('##')
	data = []
	dependency = []
	#Templates 
	template_killing  = dict()
	template_kidnap = dict()
	template_acquisition = dict()
	template_award = dict()
	template_scandal = dict()
	template_diseases = dict()
	template_disaster = dict()
	template_injury = dict()
	template_transfer = dict()
	template_phone = dict()
	instruments = ['.45-caliber semi-automatic pistol','fire_ship', 'brass_knuckles', 'four-pounder', 'Greek_fire', 'battery', 'projectile', 'stun_gun', 'sword', 'slasher', 'blade', 'steel', 'knuckles', 'W.M.D.', 'knuckle_duster', 'gun', 'field_gun', 'hatchet', 'sling', 'light_arm', 'bow', 'knife', 'brass_knucks', 'knucks', 'WMD', 'bow_and_arrow', 'stun_baton', 'flamethrower', 'shaft', 'weapon_of_mass_destruction', 'pike', 'brand', 'field_artillery', 'tomahawk', 'lance', 'cannon', 'spear', 'missile']
	product = ['market-analytics', 'software', 'restaurant', 'operating system Linux', 'music', 'cloud computing']
	fields = ['best', 'innovation', 'computer science', 'football', 'actor', 'acting']
	accusation = ['sexual','harassment', 'affair']
	injury = ['ruptured', 'knee', 'calf', 'injury', 'hamstring','cruciate', 'ligament','anterior ']
	disaster = ['flood','floods','earthquake','earthquakes','hurricane','hurricanes','storms','storm','landslide','landslides','avalanche','avalanches','tsunami','blizzard','tornado','tornadoes','drought','droughts','thunderstorm','thunderstorms']
	disease = ['cholera', 'Flu','infectious', 'diarrheal' ,'illness']
	entity_instruments = EntityMatcher(nlp, instruments, 'INSTRUMENT')
	entity_product = EntityMatcher(nlp, product, 'PRODUCT')
	entity_field = EntityMatcher(nlp, fields, 'FIELD')
	entity_accuse = EntityMatcher(nlp, accusation, 'ACCUSE')
	entity_disaster = EntityMatcher(nlp, disaster, 'DISASTER')
	entity_injury = EntityMatcher(nlp, injury, 'INJURY')
	entity_disease = EntityMatcher(nlp, disease, 'DISEASE')
	nlp.add_pipe(entity_instruments)
	#nlp.add_pipe(entity_product)
	#nlp.add_pipe(entity_field)
	#nlp.add_pipe(entity_accuse)
	#nlp.add_pipe(entity_disaster)
	#nlp.add_pipe(entity_injury)
	#nlp.add_pipe(entity_disease) 
	id = 0
	for article in articles:
		#killing-template
		killing_victim = set()
		killing_perpetrator = set()
		killing_location = set()
		killing_instrument = set()
		killing_date = set()
		#kidnap-template
		kidnap_victim = set()
		kidnap_perpetrator = set()
		kidnap_location = set()
		kidnap_date = set()
		kidnap_ransom = set()
		#acquisition-template
		acq_buyer = set()
		acq_seller = set()
		acq_product = set()
		acq_price = set()
		#award-template
		award_name = set()
		receipient = set()
		field = set()
		award_date = set()
		#scandal-template
		scandal_accused = set()
		scandal_victim = set()
		scandal_accusation = set()
		scandal_date = set()
		#diseases-template
		disease_name = set()
		disease_location = set()
		disease_casuality = set()
		disease_cause = set()
		disease_date = set()
		#disaster-template
		disaster_type = set()
		disaster_location = set()
		disaster_country = set()
		disaster_date = set()
		#injury-template
		player_name = set()
		team = set()
		injury_type = set()
		injury_time = set()
		#transfer-template
		transfer_player = set()
		transfer_team = set()
		transfer_price = set()
		transfer_time = set()
		#phone-template
		phone_company = set()
		phone_model = set()
		phone_date = set()
		phone_location = set()
		id += 1
		sentences = nltk.sent_tokenize(article)
		for sentence in sentences:
			doc = nlp(sentence)
			tokenList = nltk.word_tokenize(sentence)
			killing_person = set()
			kidnap_person = set()
			scandal_person = set()
			acq_org = set()
			date = set()
			location = set()
			ransom = set()
			for word in tokenList:

				#killing template
				if word in synonyms['killing']:
					#print(sentence)
					for ent in doc.ents:
						if ent.label_ == "PERSON":
							killing_person.add(ent.text)
						if ent.label_ == "DATE":
							killing_date.add(ent.text)
						if ent.label_ == "ORG" or ent.label_ == "GPE":
							killing_location.add(ent.text)
						if ent.label_ == "INSTRUMENT":
							killing_instrument.add(ent.text)
						# print(ent.text, ent.label_)
					
					if(re.search('(\S+\s+|^)(\S+\s+|)(kills|for killing|was killed|killing|killing of)(\s+\S+|)(\s+\S+|$)', sentence)):
						s = re.search('(\S+\s+|^)(\S+\s+|)(kills|for killing|was killed|killing|killing of)(\s+\S+|)(\s+\S+|$)', sentence).group(0)
						k = nlp(s)

						for seq in k.ents:
							if seq.label_ == 'PERSON':
								for p in killing_person:
									if seq.text in p:
										killing_victim.add(p)
					if(re.search('(\S+\s+|^)(\S+\s+|)(shot|by|fired by|killed himself|killing by|killed by| was killed by)(\s+\S+|)(\s+\S+|$)', sentence)):
						s = re.search('(\S+\s+|^)(\S+\s+|)(shot|by|fired by|killed himself|killing by|killed by| was killed by)(\s+\S+|)(\s+\S+|$)', sentence).group(0)
						k = nlp(s)
						for seq in k.ents:
							if seq.label_ == 'PERSON':
								for p in killing_person:
									if seq.text in p:
										killing_perpetrator.add(p)
				# kidnap template
				if word in synonyms['kidnap']:
					for ent in doc.ents:	
						if ent.label_ == "MONEY":
							s = re.search('[$]'+ ent.text, sentence).group(0)
							kidnap_ransom.add(s)
						if ent.label_ == "PERSON":
							kidnap_person.add(ent.text)
						if ent.label_ == "GPE" or ent.label_ == "NORP":
							kidnap_location.add(ent.text)
						if ent.label_ == "DATE":
							kidnap_date.add(ent.text)
					if(re.search('(\S+\s+|^)(\S+\s+|)(was held hostage|was kidnapped|was taken|kidnapping| kidnapping a)(\S+\s+|)', sentence)):
						s = re.search('(\S+\s+|^)(\S+\s+|)(was kidnapped|was taken|kidnapping| kidnapping a)(\S+\s+|)', sentence).group(0)
						k = nlp(s)
						for seq in k.ents:
							if seq.label_ == 'PERSON':
								for p in kidnap_person:
									if seq.text in p:
										kidnap_victim.add(p)
						for p in kidnap_person:
							if p not in kidnap_victim:
								kidnap_perpetrator.add(p)
					

				# Acquisition Template
				if word in synonyms['acquisition']:
					for ent in doc.ents:	
						if ent.label_ == "MONEY":
							acq_price.add(ent.text)
						if ent.label_ == "ORG":
							acq_org.add(ent.text)
						if ent.label_ =="PRODUCT":
							acq_product.add(ent.text)
						if(re.search('\w*\s(is selling|is acquired|is bought)\s\w*',sentence)):
							s = re.search('\w*\s(is selling|is acquired|is bought)',sentence).group(0)
							k = nlp(s)
							for seq in k.ents:
								if seq.label_ == 'ORG':
									for p in acq_org:
										if seq.text in p:
											acq_seller.add(p)
							for p in acq_org:
								if p not in acq_seller:
									acq_buyer.add(p)
						
						
				#Award Template
				if word in synonyms['award']:
					for ent in doc.ents:
						if ent.label_ == "DATE":
							award_date.add(ent.text)
						if ent.label_ == "EVENT":
							award_name.add(ent.text)
						if ent.label_ == "FIELD":
							field.add(ent.text)
						if ent.label_ == "PERSON":
							receipient.add(ent.text)
						
				#scandal template
				if word in synonyms['scandals']:
					for ent in doc.ents:
						if ent.label_ == 'DATE':
							scandal_date.add(ent.text)
						if ent.label_ == "PERSON":
							scandal_person.add(ent.text)
						if ent.label_ == "ACCUSE":
							scandal_accusation.add(ent.text)
						
						if(re.search('(\S+\s+|^)(\S+\s+|)(is accused|was accused)(\s+\S+|)(\s+\S+|$)', sentence)):
							s = re.search('(\S+\s+|^)(\S+\s+|)(is accused|was accused)(\s+\S+|)(\s+\S+|$)', sentence).group(0)
							k = nlp(s)

							for seq in k.ents:
								if seq.label_ == 'PERSON':
									for p in scandal_person:
										if seq.text in p:
											scandal_accused.add(p)
									for p in scandal_person:
										if p not in scandal_accused:
											scandal_victim.add(p)
				if word in synonyms['disaster']:
					for ent in doc.ents:
						if ent.label_ == 'DATE':
							disaster_date.add(ent.text)
						if ent.label_ == "LOC":
							disaster_location.add(ent.text)
						if ent.label_ == "GPE":
							disaster_country.add(ent.text)
						if ent.label_ == "DISASTER":
							disaster_type.add(ent.text)

						
				if word in synonyms['injury']:
					for ent in doc.ents:
						if ent.label_ == "PERSON":
							player_name.add(ent.text)
						if ent.label_ == "INJURY":
							injury_type.add(ent.text)
						if ent.label_ == "ORG":
							team.add(ent.text)
						if ent.label_ == "DATE":
							injury_time.add(ent.text)
						
				if word in synonyms['diseases']:
					for ent in doc.ents:
						if ent.label_ == "DISEASE":
							disease_name.add(ent.text)
							disease_cause.add(ent.text)
						if ent.label_ == "GPE":
							disease_location.add(ent.text)
						if ent.label_ == "CARDINAL":
							disease_casuality.add(ent.text) 
						if ent.label == "DATE":
							disease_date.add(ent.text)

			template_killing[id] = ({
				'Victim' : ' '.join(killing_victim),
				'perpetrator': ' '.join(killing_perpetrator),
				'Location' : ' '.join(killing_location),
				'Instrument': ' '.join(killing_instrument),
				'Date' : ' '.join(killing_date)
				})			
			template_kidnap[id] = ({
				'Victim' : ' '.join(kidnap_victim),
				'perpetrator': ' '.join(kidnap_perpetrator),
				'Location' : ' '.join(kidnap_location),
				'Date' : ' '.join(kidnap_date),
				'Ransom':' '.join(kidnap_ransom)
				})	
			template_acquisition[id] = ({
				'Buyer' : ' '.join(acq_buyer),
				'Seller': ' '.join(acq_seller),
				'Product' : ' '.join(acq_product),
				'Price' : ' '.join(acq_price)
				})
			template_award[id] = ({
				'Award_Name' : ' '.join(award_name),
				'Receipient': ' '.join(receipient),
				'Field' : ' '.join(field),
				'Date' : ' '.join(award_date)
				})	
			template_scandal[id] = ({
				'Accused' : ' '.join(scandal_accused),
				'Victim': ' '.join(scandal_victim),
				'Accusation' : ' '.join(scandal_accusation),
				'Date' : ' '.join(scandal_date)
				})
			template_transfer[id] = ({
				'Victim' : ' '.join(kidnap_victim),
				'perpetrator': ' '.join(kidnap_perpetrator),
				'Location' : ' '.join(kidnap_location),
				'Date' : ' '.join(kidnap_date),
				'Ransom':' '.join(kidnap_ransom)
				})	
			template_injury[id] = ({
				'Player_name' : ' '.join(player_name),
				'Team': ' '.join(team),
				'Injury_type' : ' '.join(injury_type),
				'Time_to_recover' : ' '.join(injury_time)
				})
			template_phone[id] = ({
				'Victim' : ' '.join(kidnap_victim),
				'perpetrator': ' '.join(kidnap_perpetrator),
				'Location' : ' '.join(kidnap_location),
				'Date' : ' '.join(kidnap_date),
				'Ransom':' '.join(kidnap_ransom)
				})	
			template_disaster[id] = ({
				'Type' : ' '.join(disaster_type),
				'Location' : ' '.join(disaster_location),
				'Country' : ' '.join(disaster_country),
				'Date': ' '.join(disaster_date)
				})
			template_diseases[id] = ({
				'Name' : ' '.join(disease_name),
				'Location': ' '.join(disease_location),
				'Casualities' : ' '.join(disease_casuality),
				'Causes' : ' '.join(disease_cause),
				'Date':' '.join(disease_date)
				})							
			for token in doc:
				#print(token)
				data.append({
					"token": token.text,
					"lemma": token.lemma_,
					"pos": token.pos_ ,
					"tag": token.tag_ ,
					"dependency": token.dep_ ,
					"hypernyms":getHypernyms(str(token)), 
					"holonyms" :getHolonyms(str(token)) ,
					"meronyms" :getMeronyms(str(token)) ,
					"hyponyms" : getHyponyms(str(token))
					})
				dependency.append({
					"Text" :token.text,
					"dependency": token.dep_,
					"Head Text" : token.head.text,
					"Head Pos" : token.head.pos_ ,
					"children" : [child for child in token.children]
					})
				#print(dependency)
				#print(data)
	print(template_killing)
	#print(template_kidnap)
	#print(template_acquisition)
	#print(template_award)
	#print(template_scandal)
	#print(template_disaster)
	#print(template_injury)
	#print(template_diseases)
	#print(template_transfer)
	#print(template_phone)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = "abc"  

ALLOWED_EXTENSIONS = set(['txt'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/',methods=['GET', 'POST'])
def Home():
    return render_template('home.html')


@app.route('/upload',methods=['POST'])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    now = datetime.now()
    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('File successfully uploaded ' + file.filename + ' to the database!')
    else:
        print('Invalid Upload only excel file') 
    msg = 'Success Upload'
    session['file']=os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template('home.html')

@app.route('/home_result',methods=['POST'])
def home_result():   
    filename = session['file']
    synonyms = {
	'killing' : {'shot','fired','killed','murdered', 'murder', 'killing', 'vote_out', 'cleanup', 'defeat', 'vote_down', 'pour_down', 'drink_down', 'kill', 'obliterate', 'wipe_out', 'sidesplitting', 'pop', 'down', 'toss_off', 'belt_down', 'shoot_down', 'bolt_down', 'violent_death', 'stamp_out', 'putting_to_death'},
	'kidnap' : {'kidnappers', 'kidnapping', 'forcefully','kidnap', 'abduct', 'capture','seize', 'snatch', 'kidnapped', 'abducted', 'captured', 'snatched','hostage', 'ransom', 'taken', 'missing', 'disappearance', 'vanished', 'disappeared'},
	'acquisition' : {'bought','buy','purchased','acquired','acquire','purchase','takeover','merger','selling', 'sold', 'sell'},
	'award' : {"award", "awarded", "Award", "Awarded", "wins", "honored"},
	'scandals' :{"harassment", "affair", "sexual", "rumour", "rumoured", "accused", "accuse", "accuses"},
	'disaster' : {'flood','floods','earthquake','earthquakes','hurricane','hurricanes','storms','storm','landslide','landslides','avalanche','avalanches','tsunami','blizzard','tornado','tornadoes','drought','droughts','thunderstorm','thunderstorms'},
	'injury' : {'injury','injured','injuries', 'ruled', 'out'},
	'diseases':{'outbreak','disease','widespread', 'epidemic', 'pandemic', 'spread', 'deaths'}
	}
	nlp = spacy.load('en_core_web_sm') 
    # filename = 'corpus.txt'
	#filename = "The bullet responsible for killing Ron Helus from Ventura County during November's mass shooting at the Borderline Bar & Grill was fired by Ian David Long, authorities said Friday."
	file = open(filename, 'r', encoding="utf-8")
	document = file.read()
	articles = document.split('##')
	data = []
	dependency = []
	#Templates 
	template_killing  = dict() 
    instruments = ['.45-caliber semi-automatic pistol','fire_ship', 'brass_knuckles', 'four-pounder', 'Greek_fire', 'battery', 'projectile', 'stun_gun', 'sword', 'slasher', 'blade', 'steel', 'knuckles', 'W.M.D.', 'knuckle_duster', 'gun', 'field_gun', 'hatchet', 'sling', 'light_arm', 'bow', 'knife', 'brass_knucks', 'knucks', 'WMD', 'bow_and_arrow', 'stun_baton', 'flamethrower', 'shaft', 'weapon_of_mass_destruction', 'pike', 'brand', 'field_artillery', 'tomahawk', 'lance', 'cannon', 'spear', 'missile']  
    entity_instruments = EntityMatcher(nlp, instruments, 'INSTRUMENT')
    nlp.add_pipe(entity_instruments)
    id = 0
	for article in articles:
		#killing-template
		killing_victim = set()
		killing_perpetrator = set()
		killing_location = set()
		killing_instrument = set()
		killing_date = set()
        id += 1
		sentences = nltk.sent_tokenize(article)
		for sentence in sentences:
			doc = nlp(sentence)
			tokenList = nltk.word_tokenize(sentence)
			killing_person = set()
			kidnap_person = set()
			scandal_person = set()
			acq_org = set()
			date = set()
			location = set()
			ransom = set()
			for word in tokenList:

				#killing template
				if word in synonyms['killing']:
					#print(sentence)
					for ent in doc.ents:
						if ent.label_ == "PERSON":
							killing_person.add(ent.text)
						if ent.label_ == "DATE":
							killing_date.add(ent.text)
						if ent.label_ == "ORG" or ent.label_ == "GPE":
							killing_location.add(ent.text)
						if ent.label_ == "INSTRUMENT":
							killing_instrument.add(ent.text)
						# print(ent.text, ent.label_)
					
					if(re.search('(\S+\s+|^)(\S+\s+|)(kills|for killing|was killed|killing|killing of)(\s+\S+|)(\s+\S+|$)', sentence)):
						s = re.search('(\S+\s+|^)(\S+\s+|)(kills|for killing|was killed|killing|killing of)(\s+\S+|)(\s+\S+|$)', sentence).group(0)
						k = nlp(s)

						for seq in k.ents:
							if seq.label_ == 'PERSON':
								for p in killing_person:
									if seq.text in p:
										killing_victim.add(p)
					if(re.search('(\S+\s+|^)(\S+\s+|)(shot|by|fired by|killed himself|killing by|killed by| was killed by)(\s+\S+|)(\s+\S+|$)', sentence)):
						s = re.search('(\S+\s+|^)(\S+\s+|)(shot|by|fired by|killed himself|killing by|killed by| was killed by)(\s+\S+|)(\s+\S+|$)', sentence).group(0)
						k = nlp(s)
						for seq in k.ents:
							if seq.label_ == 'PERSON':
								for p in killing_person:
									if seq.text in p:
										killing_perpetrator.add(p)
            template_killing[id] = ({
				'Victim' : ' '.join(killing_victim),
				'perpetrator': ' '.join(killing_perpetrator),
				'Location' : ' '.join(killing_location),
				'Instrument': ' '.join(killing_instrument),
				'Date' : ' '.join(killing_date)
				})	
    
    
    ans = []
    region = {'APAC': 0, 'EMEA': 1}
    entity = {'SSB FFT': 0, 'SSB HK': 1, 'SSB LON': 2, 'SSB PAR': 3, 'SSB TOK': 4}
    counterparty_type = {'Bank': 0, 'Corporate': 1}
    counterparty_name = {'ANZ': 0, 'Allianz': 1, 'BNP': 2, 'Barclays': 3, 'CACIB': 4, 'Citibank HK': 5, 'Commerzbank': 6, 'Credit Suisse': 7, 'Deutsche': 8, 'HSBC': 9, 'Mitsubishi': 10, 'Shell': 11, 'Toyota': 12, 'Volkswagen': 13}
    product = {'Fwd': 0, 'NDF': 1, 'NDO': 2, 'NDS': 3, 'Option': 4, 'Spot': 5, 'Swap': 6, 'Time Option': 7}
    ccy_pair = {'AUDJPY': 0, 'AUDNZD': 1, 'AUDUSD': 2, 'EURCAD': 3, 'EURCHF': 4, 'EURGBP': 5, 'EURHKD': 6, 'EURHUF': 7, 'EURJPY': 8, 'EURNOK': 9, 'EURNZD': 10, 'EURSGD': 11, 'EURTRY': 12, 'EURUSD': 13, 'GBPCHF': 14, 'GBPJPY': 15, 'GBPUSD': 16, 'NZDJPY': 17, 'NZDUSD': 18, 'USDCAD': 19, 'USDCHF': 20, 'USDCNY': 21, 'USDDKK': 22, 'USDHKD': 23, 'USDHUF': 24, 'USDIDR': 25, 'USDINR': 26, 'USDJPY': 27, 'USDKRW': 28, 'USDKZT': 29, 'USDNGN': 30, 'USDNOK': 31, 'USDRUB': 32, 'USDSGD': 33, 'USDTRY': 34, 'USDTWD': 35, 'USDUGX': 36}
    broker = {'CMC': 0, 'IG': 1, 'Interactive': 2, 'LCG': 3, 'LCH': 4, 'Pepperstone': 5, 'Saxo': 6, 'Swissquote': 7, 'TD': 8}
    tenor = {'1M': 0, '1W': 1, '1Y': 2, '2M': 3, '2W': 4, '2Y': 5, '3M': 6, '3W': 7, '4M': 8, '5M': 9, '6M': 10, '7M': 11, 'S': 12}
    ent_reg = {'SSB HK': 'APAC', 'SSB TOK': 'APAC', 'SSB LON': 'EMEA', 'SSB FFT': 'EMEA', 'SSB PAR': 'EMEA'}
    cp_cpt = {'Citibank HK': 'Bank', 'Mitsubishi': 'Corporate', 'Toyota': 'Corporate', 'ANZ': 'Bank', 'Shell': 'Corporate', 'Barclays': 'Bank', 'HSBC': 'Bank', 'Deutsche': 'Bank', 'Volkswagen': 'Corporate', 'Commerzbank': 'Bank', 'Allianz': 'Corporate', 'Credit Suisse': 'Bank', 'BNP': 'Bank', 'CACIB': 'Bank'}
    list = [region, entity, counterparty_name, counterparty_type, product, ccy_pair, tenor]
    encoder = {0:encoder1, 1:encoder2, 2:encoder3, 3:encoder4, 4:encoder5}
    scaler = {0:scaler1, 1:scaler2, 2:scaler3, 3:scaler4, 4:scaler5}
    models = {0:model1, 1:model2, 2:model3, 3:model4, 4:model5}
    for index, x in df.iterrows():
            x_test=[[i for i in x]]
            region = ent_reg[x_test[0][0]]
            cp_type = cp_cpt[x_test[0][1]]
            x_test[0].insert(0,region)
            x_test[0].insert(3,cp_type)
            # print(x_test)
            for i in range(0, 7):
                x_test[0][i] = list[i][x_test[0][i]]
            x_test[0][7] = int(x_test[0][7])
            x_test[0][8] = float(x_test[0][8])
            #print(x_test)
            t = joblib.load('minmax_scaler_us1.pkl')
            cluster = kmeans.predict(t.transform(x_test))
            #print(cluster)
            encoded_arr = encoder[cluster[0]].predict(scaler[cluster[0]].transform(x_test))
            prediction = models[cluster[0]].predict(encoded_arr)
            output = prediction[0]
            if(output == 1):
                ans.append("Detected")
            else:
                ans.append("Not Detected")
    return render_template('home_result.html', result = ans,dataset=df)

@app.route('/explain/<id>',methods=['GET','POST'])
def explainid(id):
    with open('../Final Architecture/Pickled Models/explainer_segment_1.pkl', 'rb') as f: explainer_seg_1 = dill.load(f)
    with open('../Final Architecture/Pickled Models/explainer_segment_2.pkl', 'rb') as f: explainer_seg_2 = dill.load(f)
    with open('../Final Architecture/Pickled Models/explainer_segment_3.pkl', 'rb') as f: explainer_seg_3 = dill.load(f)
    with open('../Final Architecture/Pickled Models/explainer_segment_4.pkl', 'rb') as f: explainer_seg_4 = dill.load(f)
    with open('../Final Architecture/Pickled Models/explainer_segment_5.pkl', 'rb') as f: explainer_seg_5 = dill.load(f)
    file=session['file']
    id=int(id)
    data = {}
    df = pd.read_excel(file)
    region = {'APAC': 0, 'EMEA': 1}
    entity = {'SSB FFT': 0, 'SSB HK': 1, 'SSB LON': 2, 'SSB PAR': 3, 'SSB TOK': 4}
    counterparty_type = {'Bank': 0, 'Corporate': 1}
    counterparty_name = {'ANZ': 0, 'Allianz': 1, 'BNP': 2, 'Barclays': 3, 'CACIB': 4, 'Citibank HK': 5, 'Commerzbank': 6, 'Credit Suisse': 7, 'Deutsche': 8, 'HSBC': 9, 'Mitsubishi': 10, 'Shell': 11, 'Toyota': 12, 'Volkswagen': 13}
    product = {'Fwd': 0, 'NDF': 1, 'NDO': 2, 'NDS': 3, 'Option': 4, 'Spot': 5, 'Swap': 6, 'Time Option': 7}
    ccy_pair = {'AUDJPY': 0, 'AUDNZD': 1, 'AUDUSD': 2, 'EURCAD': 3, 'EURCHF': 4, 'EURGBP': 5, 'EURHKD': 6, 'EURHUF': 7, 'EURJPY': 8, 'EURNOK': 9, 'EURNZD': 10, 'EURSGD': 11, 'EURTRY': 12, 'EURUSD': 13, 'GBPCHF': 14, 'GBPJPY': 15, 'GBPUSD': 16, 'NZDJPY': 17, 'NZDUSD': 18, 'USDCAD': 19, 'USDCHF': 20, 'USDCNY': 21, 'USDDKK': 22, 'USDHKD': 23, 'USDHUF': 24, 'USDIDR': 25, 'USDINR': 26, 'USDJPY': 27, 'USDKRW': 28, 'USDKZT': 29, 'USDNGN': 30, 'USDNOK': 31, 'USDRUB': 32, 'USDSGD': 33, 'USDTRY': 34, 'USDTWD': 35, 'USDUGX': 36}
    broker = {'CMC': 0, 'IG': 1, 'Interactive': 2, 'LCG': 3, 'LCH': 4, 'Pepperstone': 5, 'Saxo': 6, 'Swissquote': 7, 'TD': 8}
    tenor = {'1M': 0, '1W': 1, '1Y': 2, '2M': 3, '2W': 4, '2Y': 5, '3M': 6, '3W': 7, '4M': 8, '5M': 9, '6M': 10, '7M': 11, 'S': 12}
    ent_reg = {'SSB HK': 'APAC', 'SSB TOK': 'APAC', 'SSB LON': 'EMEA', 'SSB FFT': 'EMEA', 'SSB PAR': 'EMEA'}
    cp_cpt = {'Citibank HK': 'Bank', 'Mitsubishi': 'Corporate', 'Toyota': 'Corporate', 'ANZ': 'Bank', 'Shell': 'Corporate', 'Barclays': 'Bank', 'HSBC': 'Bank', 'Deutsche': 'Bank', 'Volkswagen': 'Corporate', 'Commerzbank': 'Bank', 'Allianz': 'Corporate', 'Credit Suisse': 'Bank', 'BNP': 'Bank', 'CACIB': 'Bank'}
    list = [region, entity, counterparty_name, counterparty_type, product, ccy_pair, tenor]
    data = {"list": list}
    encoder = {0:encoder1, 1:encoder2, 2:encoder3, 3:encoder4, 4:encoder5}
    scaler = {0:scaler1, 1:scaler2, 2:scaler3, 3:scaler4, 4:scaler5}
    models = {0:model1, 1:model2, 2:model3, 3:model4, 4:model5}
    explainer_ds= {0:explainer_seg_1, 1:explainer_seg_2, 2:explainer_seg_3, 3:explainer_seg_4, 4:explainer_seg_5}
    t = joblib.load('minmax_scaler_us1.pkl')
    x_test=[[i for i in df.iloc[id]]]
    region = ent_reg[x_test[0][0]]
    cp_type = cp_cpt[x_test[0][1]]
    x_test[0].insert(0,region)
    x_test[0].insert(3,cp_type)
    p=[i for i in x_test[0]]
    n=len(p)
    name=['Region','Entity','Counterparty','Product_Type','Ccy_pairs','Tenors','Amt','Rate']
    new_name=['Region','Entity','Counterparty','Product Type','Currency Pair','Tenor','Amount (in Millions)','Rate']
    feature=set()
    # print(x_test)
    for i in range(0, 7):
        x_test[0][i] = list[i][x_test[0][i]]
    x_test[0][7] = int(x_test[0][7])
    x_test[0][8] = float(x_test[0][8])
    #print(x_test)
    cluster = kmeans.predict(t.transform(x_test))
    #print(cluster)
    encoded_arr = encoder[cluster[0]].predict(scaler[cluster[0]].transform(x_test))
    prediction = models[cluster[0]].predict(encoded_arr)
    output = prediction[0]
    #exp = exp.as_html()
    data["cluster"] = cluster[0]
    if(output == 1):
        data["output"] = "Detected"
        x_test=pd.Series(x_test[0])
        #print(x_test)
        explainer = explainer_ds[cluster[0]]
        predict_forest = lambda x: models[cluster[0]].predict_proba(encoder[cluster[0]].predict(scaler[cluster[0]].transform(x)))
        # print(predict_forest([x_test]))
        exp = explainer.explain_instance(x_test, predict_forest, top_labels=1)
        val=exp.as_list()
        print(val)
        d=dict()
        for i in val:
            s=re.sub('[^A-Za-z_]', '', i[0])
            d[s]=i[1]
        for i in d.keys():
            if d[i]>0:
                if i == 'Counterparty_Type' or i=='Region' or i=='Entity':
                    feature.add('Counterparty')
                    continue
                feature.add(i)
    else:
        data["output"] = "Not Detected"
    css_link1 = "../static/developerPg.css"
    css_link2 = "../static/form_style.css"
    p.pop(3)
    return render_template('explain.html', data = data,p=p,features=feature,name=name,new_name=new_name,n=n,css_link1= css_link1, css_link2= css_link2)

@app.route('/explain',methods=['POST'])
def explain():
    return render_template('explain.html',id=id)

if __name__=='__main__':
    app.run(host='0.0.0.0')