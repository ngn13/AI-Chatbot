BOT_TOKEN = "" # Your bot token here.

from discord.ext.commands.errors import MissingRequiredArgument


try:
    from discord.channel import DMChannel
    import nltk
    from nltk.util import pr
    from nltk.stem.lancaster import LancasterStemmer
    print('IMPORTANT: If this is the first time you are running, you may need to install punkt using "nltk.download("punkt")."')
    #nltk.download('punkt') # If this your first time running, you may need this line.

    stemmer = LancasterStemmer()

    import requests
    import sys
    import datetime
    import numpy
    import tflearn
    import tensorflow
    import random
    import json
    import pickle
    import os
    import discord
    from discord.ext import commands
    from discord.ext.commands import has_permissions, MissingPermissions
except Exception as e:
    print('Ops! Got an import error: {e}\nMake sure you have installed all the dependencies, check out https://github.com/ngn13/chatx for more information.')
    exit()

try:
    ARGV1 = sys.argv[1]
    if ARGV1 != "True" and ARGV1 != "False":
        print('Unknown command line argument for model. Using default: "False"')
        ARGV1 = "False"
except:
    print('No command line argument specified for model. Using default: "False".')
    ARGV1 = "False"

try:
    ARGV2 = sys.argv[2]
    if ARGV2 != "True" and ARGV2 != "False":
        print('Unknown command line argument for pickle data. Using default: "False"')
        ARGV2 = "False"
except:
    print('No command line argument specified for pickle data. Using default: "False".')
    ARGV2 = "False"

with open('intents.json') as file:
    data = json.load(file)

if ARGV2 == "False":
    try:
        with open('./pickle/data.pickle', 'rb') as file:
            words, labels, training, output = pickle.load(file)
    except:
        print('Error: No pickle file found. Check https://github.com/ngn13/chatx for more information.')
        exit()

else:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            words1 = nltk.word_tokenize(pattern)
            words.extend(words1)
            docs_x.append(words1)
            docs_y.append(intent['tag'])

            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        words1 = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in words1:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open('./pickle/data.pickle', 'wb') as file:
        pickle.dump((words, labels, training, output), file)

tensorflow.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

if ARGV1 == "False":
    try:
        model.load('./model/model.tflearn')
    except:
        print('Error: No model found. Check https://github.com/ngn13/chatx for more information.')
        exit()
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('./model/model.tflearn')

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w.lower()) for w in s_words]

    for sw in s_words:
        for i, w in enumerate(words):
            if w == sw:
                bag[i] = 1
    
    return numpy.array(bag)

def predict(msg):
    pred = model.predict([bag_of_words(msg, words)])[0]

    pred_index = numpy.argmax(pred)
    tag = labels[pred_index]
    if pred[pred_index] > 0.7:
        for tg in data['intents']:
            if tg['tag'] == tag:
                foundtag = tg['tag']
                responses = tg['responses']
        return [random.choice(responses), foundtag]

    else:
        return ["**Sorry**, I couldn't understand :/", 'null']

async def make_act(message, pred):
    if pred[0] == "{DATE}":
        await message.reply(f'According to my local machine it is: **{datetime.date.today().strftime("%B %d, %Y")}**')
    elif pred[0] == "{LEAVE-ACT}":
        n = random.randint(0,1000)
        if n == 42:
            await message.reply('OK, then i will leave.')
            await message.guild.leave()
        else:
            await message.reply("**Sorry** :( I am trying my best.")
    elif pred[0] == "{REACT-LOVE}":
        emoji = discord.utils.get(message.guild.emojis, name='heart')
        await message.add_reaction('❤️')
    elif pred[0] == "{JOKE}":
        try:
            r = requests.get(url='https://icanhazdadjoke.com/', headers={"Accept": "application/json"})
            data = r.json()
            await message.reply(str(data['joke']))
        except Exception as e:
            print(f'Ops! Cannot connect jokes API, reason: {e}')
            await message.reply("I don't have any jokes right now. Ask me later.")
    elif pred[0] == "{AMAZING-FACT}":
        try:
            limit = 1
            api_url = 'https://api.api-ninjas.com/v1/facts?limit={}'.format(limit)
            r = requests.get(api_url, headers={'X-Api-Key': 'ZXvwkmQtLHEjrYdDc9cyGQ==FvkFP3A67gk9PMQu'})
            data = r.json()
            await message.reply(str(data[0]['fact']))
        except Exception as e:
            print(f'Ops! Cannot connect random facts API, reason: {e}')
            await message.reply("I don't have any facts right now. Ask me later.")
    elif pred[0] == "{PP}":
        pp = message.author.avatar_url
        embedVar = discord.Embed(title="Here is your profile picture!", color=0x00E999B)
        embedVar.set_image(url=(pp))
        await message.reply(embed=embedVar)
    else:
        await message.reply(pred[0])

client = discord.Client()

client = commands.Bot(command_prefix=">>>", help_command=None)

def saveDB():
    # This function will be called after every change on the list called DB
    # So if you have a real database you can add a save function or something
    pass

DB = []
halfActivePreds = ['greeting', 'goodbye'] # If you want to a tag to be a part of half active mode, add it here

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event 
async def on_guild_join(guild):
    for c in guild.text_channels:
        try:
            embedVar = discord.Embed(title="<3", description=f"Thanks for adding me to your server.\nMy prefix is: `>>>`\nHelp command: `help`", color=0x00E999B)
            await c.send(embed=embedVar)
            break
        except:
            pass

@client.event
async def on_guild_remove(guild):
    for entry in DB:
        serverid, channelid, mode = entry
        if serverid == guild.id:
            DB.remove(entry)
            saveDB()

@client.event
async def on_command_error(ctx, error):
    if isinstance(error, MissingPermissions):
        await ctx.channel.send("Missing permissions!")
    if isinstance(error, MissingRequiredArgument):
        await ctx.channel.send("Ops! Missing required argument! Use command `help` to get help.")
    else:
        print(f'Bot error: {error}')

@client.command()
@has_permissions(manage_guild=True)  
async def setChannel(ctx, channel : discord.TextChannel, *, mode):
    if mode == "active" or mode == "ping" or mode == "half active" or mode == "default" or mode == "disabled":
        if mode == "disabled":
            for entry in DB:
                serverid, chnlid, modetmp = entry
                if serverid == ctx.guild.id and chnlid == channel.id:
                    DB.remove(entry)
        else:
            for entry in DB:
                serverid, chnlid, modetmp = entry
                if serverid == ctx.guild.id and chnlid == channel.id:
                    DB.remove(entry)
            DB.append((ctx.guild.id, channel.id, mode))
        embedVar = discord.Embed(title="Sucess!", description=f"Set the mode `{mode}` for the {channel.mention}.", color=0x00E999B)
        await ctx.channel.send(embed=embedVar)
        saveDB()
    else:
        embedVar = discord.Embed(title="Error!", description="Unknown mode. Use the command `help` to see mods.", color=0x00E999B)
        await ctx.channel.send(embed=embedVar)

@client.command()
async def help(ctx):
    with open('help.json', 'r') as helpf:
        helpc = json.load(helpf)
    embedVar = discord.Embed(title="Help Menu", color=0x00E999B)
    for c in helpc:
        embedVar.add_field(name=c['command'], value=c['desc'], inline=False)
    
    await ctx.channel.send(embed=embedVar)


@client.event
async def on_message(message):
    await client.process_commands(message)
    if message.channel != DMChannel:
        if message.author != client.user:
            for entry in DB:
                serverid, channelid, mode = entry
                if message.guild.id == serverid and channelid == message.channel.id:
                    if mode == "active":
                        prediction = predict(message.content.lower())
                        await make_act(message, prediction)
                    
                    elif mode == "ping":
                        if client.user.mentioned_in(message):
                            newmessage = message.content.replace(f'@{client.user}', ' ')
                            prediction = predict(newmessage.lower())
                            await make_act(message, prediction)
                    
                    elif mode == "half active":
                        prediction = predict(message.content.lower())
                        if prediction[1] in halfActivePreds:
                            await make_act(message, prediction)
                    
                    elif mode == "default":
                        if client.user.mentioned_in(message):
                            newmessage = message.content.replace(f'@{client.user}', ' ')
                            prediction = predict(newmessage.lower())
                            await make_act(message, prediction)
                        else:
                            prediction = predict(message.content.lower())
                            if prediction[1] in halfActivePreds:
                                await make_act(message, prediction)

client.run(BOT_TOKEN)














