from gtts import gTTS
from playsound import playsound
import shutil
import os

path = ("reactions")
if not os.path.exists(path):
    os.mkdir(path)

#1 emo
tts_angry = gTTS("Wow you look cute even when angry! ")
tts_angry.save("anger.mp3")

tts_fear = gTTS("Don't be Scared Buddy I will protect you. ")
tts_fear.save("fear.mp3")

tts_happy = gTTS("You look preatty when you smile!")
tts_happy.save("happy.mp3")

tts_neutral = gTTS("Don't be so serious dude! ")
tts_neutral.save("neutral.mp3")

tts_sad = gTTS("Hello! There sad face I would appreciate a smile.")
tts_sad.save("sad.mp3")

tts_surprise = gTTS("You seem surprised at how awesome I am.")
tts_surprise.save("surprise.mp3")

#2
tts_loser = gTTS("Nothing wrong in being a loser. You can be a winner somebody too. ")
tts_loser.save("loser.mp3")

tts_punch = gTTS("You think that you are Bruce Lee now? ")
tts_punch.save("punch.mp3")

tts_super = gTTS("Wow, thats wonderful!")
tts_super.save("super.mp3")

tts_victory = gTTS("Yes! We are indeed victorious.")
tts_victory.save("victory.mp3")

files = ['anger.mp3', 'fear.mp3', 'happy.mp3', 'neutral.mp3', 'surprise.mp3', 'loser.mp3', 'punch.mp3', 'super.mp3', 'victory.mp3']

for f in files:
    shutil.move(f, path)

playsound("reactions/neutral.mp3")
