def enc(word,shift):
    new_words=" "
    for letter in word:
         letter=chr(ord(letter)+shift)
         new_words=new_words+letter
         
    print(new_words)


def dec(word,shift):
     new_words=" "
     for letter in word:
          letter=chr(ord(letter)-shift)
          new_words=new_words+letter
          
     print(new_words)

word_str=input("type the letter you want to encrypt or decrypt")
wanting=input("you want to encrypt(0) or decrypt(1)")
shiftstr=input("how much shift do you want")
shift=int(shiftstr)
word=word_str.lower()
if wanting=="1":
     dec(word,shift)
else:
     enc(word,shift)



