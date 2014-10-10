#Agnes               en_US    # Isn't it nice to have a computer that will talk to you?
#Albert              en_US    #  I have a frog in my throat. No, I mean a real frog!
#Alex                en_US    # Most people recognize me by my voice.
#Bad News            en_US    # The light you see at the end of the tunnel is the headlamp of a fast approaching train.
#Bahh                en_US    # Do not pull the wool over my eyes.
#Bells               en_US    # Time flies when you are having fun.
#Boing               en_US    # Spring has sprung, fall has fell, winter's here and it's colder than usual.
#Bruce               en_US    # I sure like being inside this fancy computer
#Bubbles             en_US    # Pull the plug! I'm drowning!
#Cellos              en_US    # Doo da doo da dum dee dee doodly doo dum dum dum doo da doo da doo da doo da doo da doo da doo
#Deranged            en_US    # I need to go on a really long vacation.
#Fred                en_US    # I sure like being inside this fancy computer
#Good News           en_US    # Congratulations you just won the sweepstakes and you don't have to pay income tax again.
#Hysterical          en_US    # Please stop tickling me!
#Junior              en_US    # My favorite food is pizza.
#Kathy               en_US    # Isn't it nice to have a computer that will talk to you?
#Pipe Organ          en_US    # We must rejoice in this morbid voice.
#Princess            en_US    # When I grow up I'm going to be a scientist.
#Ralph               en_US    # The sum of the squares of the legs of a right triangle is equal to the square of the hypotenuse.
#Trinoids            en_US    # We cannot communicate with these carbon units.
#Vicki               en_US    # Isn't it nice to have a computer that will talk to you?
#Victoria            en_US    # Isn't it nice to have a computer that will talk to you?
#Whisper             en_US    # Pssssst, hey you, Yeah you, Who do ya think I'm talking to, the mouse?
#Zarvox              en_US    # That looks like a peaceful planet.

while read word; do
    for voice in Agnes Alex Bruce Fred Kathy Ralph Vicki Victoria; do
        #echo $word
        #echo $voice
        say $word -v $voice -o ${voice}_${word}.aif
    done
done < words.txt
