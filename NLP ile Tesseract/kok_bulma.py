listem=["efsanevi indirimler başlıyor","alışverişinizde geçerli %50 indirim","ürünlerinizi sepete eklemeyi unutmayın "]

yeniListem=[]
for i in listem:
    yeniListem.append(i.split(" "))


for j in range(len(yeniListem)):
    
    for k in range(len(yeniListem[j])):
        
        if yeniListem[j][k] == "efsanevi":
            
            yeniListem[j][k]=yeniListem[j][k].replace("efsanevi", "efsane")
            
        elif yeniListem[j][k] == "alışverişinizde":
            
            yeniListem[j][k]=yeniListem[j][k].replace("alışverişinizde", "alışveriş")
            
        elif yeniListem[j][k] == "indirimler":
            
            yeniListem[j][k]=yeniListem[j][k].replace("indirimler", "indirim")
            
        elif yeniListem[j][k] == "ürünlerinizi":
            
            yeniListem[j][k]=yeniListem[j][k].replace("ürünlerinizi", "ürün")
            
        elif yeniListem[j][k] == "sepete":
            
            yeniListem[j][k]=yeniListem[j][k].replace("sepete", "sepet")
            
        elif yeniListem[j][k] == "fiyatları":
            
            yeniListem[j][k]=yeniListem[j][k].replace("fiyatları", "fiyat")
        
        elif yeniListem[j][k] == "ürünlerinde":
            
            yeniListem[j][k]=yeniListem[j][k].replace("ürünlerinde", "ürün")
            
        elif yeniListem[j][k] == "indirimlerle":
            
            yeniListem[j][k]=yeniListem[j][k].replace("indirimlerle", "indirim")
        
        elif yeniListem[j][k] == "ürünlerde":
            
            yeniListem[j][k]=yeniListem[j][k].replace("ürünlerde", "ürün")
        
        elif yeniListem[j][k] == "üzerine":
            
            yeniListem[j][k]=yeniListem[j][k].replace("üzerine", "üzeri")
        
        elif yeniListem[j][k] == "sepette":
            
            yeniListem[j][k]=yeniListem[j][k].replace("sepette", "sepet")
        
        elif yeniListem[j][k] == "alışverişe":
            
            yeniListem[j][k]=yeniListem[j][k].replace("alışverişe", "alışveriş")
        
        elif yeniListem[j][k] == "alışverişinize":
            
            yeniListem[j][k]=yeniListem[j][k].replace("alışverişinize", "alışveriş")
        
        elif yeniListem[j][k] == "alışverişine":
            
            yeniListem[j][k]=yeniListem[j][k].replace("alışverişine", "alışveriş")
            
        elif yeniListem[j][k] == "fırsatlarla":
            
            yeniListem[j][k]=yeniListem[j][k].replace("fırsatlarla", "fırsat")
        
        elif yeniListem[j][k] == "indirimlerini":
            
            yeniListem[j][k]=yeniListem[j][k].replace("indirimlerini", "indirim")
        
        elif yeniListem[j][k] == "kaçırmayın":
            
            yeniListem[j][k]=yeniListem[j][k].replace("kaçırmayın", "kaçmaz")
        
        elif yeniListem[j][k] == "markalarında":
            
            yeniListem[j][k]=yeniListem[j][k].replace("markalarında", "marka")          
           
        
        elif yeniListem[j][k] == "kaçırmayın":
            
            yeniListem[j][k]=yeniListem[j][k].replace("kaçırmayın", "kaçırma")
        
        elif yeniListem[j][k] == "alışverişlerinde":
            
            yeniListem[j][k]=yeniListem[j][k].replace("alışverişlerinde", "alışveriş")
        
        elif yeniListem[j][k] == "keşfetmen":
            
            yeniListem[j][k]=yeniListem[j][k].replace("keşfetmen", "keşfet")
        
        elif yeniListem[j][k] == "fırsatları":
            
            yeniListem[j][k]=yeniListem[j][k].replace("fırsatları", "fırsat")
        
        elif yeniListem[j][k] == "indirimine":
            
            yeniListem[j][k]=yeniListem[j][k].replace("indirimine", "indirim")
        
        elif yeniListem[j][k] == "indirimli":
            
            yeniListem[j][k]=yeniListem[j][k].replace("indirimli", "indirim")
        
        elif yeniListem[j][k] == "ürünleri":
            
            yeniListem[j][k]=yeniListem[j][k].replace("ürünleri", "ürün")
        
        elif yeniListem[j][k] == "alışverişlerde":
            
            yeniListem[j][k]=yeniListem[j][k].replace("alışverişlerde", "alışveriş")
            
        elif yeniListem[j][k] == "alışverişe":

            yeniListem[j][k]=yeniListem[j][k].replace("alışverişe", "alışveriş")

        elif yeniListem[j][k] == "kaçırmayın":
        
                    yeniListem[j][k]=yeniListem[j][k].replace("kaçırmayın", "kaçmaz")
        
        elif yeniListem[j][k] == "ürünlerde":
        
                    yeniListem[j][k]=yeniListem[j][k].replace("ürünlerde", "ürün")
        
        elif yeniListem[j][k] == "fiyatlarla":
        
                    yeniListem[j][k]=yeniListem[j][k].replace("fiyatlarla", "fiyat")
        
        elif yeniListem[j][k] == "fırsatı":
        
                    yeniListem[j][k]=yeniListem[j][k].replace("fırsatı", "fırsat")
        
        elif yeniListem[j][k] == "mağazalarında":
        
                    yeniListem[j][k]=yeniListem[j][k].replace("mağazalarında", "mağaza")
        
        elif yeniListem[j][k] == "indirimleri":
        
                    yeniListem[j][k]=yeniListem[j][k].replace("indirimleri", "indirim")
        
        elif yeniListem[j][k] == "indirimlerle":
        
                    yeniListem[j][k]=yeniListem[j][k].replace("indirimlerle", "indirim")
        
        
        elif yeniListem[j][k] == "fırsatıyla":
        
                    yeniListem[j][k]=yeniListem[j][k].replace("fırsatıyla", "fırsat")
        
        
        elif yeniListem[j][k] == "fiyatına":
        
                    yeniListem[j][k]=yeniListem[j][k].replace("fiyatına", "fiyat")
                    
                    
        elif yeniListem[j][k] == "kampanyalardan":
            yeniListem[j][k]=yeniListem[j][k].replace("kampanyalardan", "kampanya")
            

        elif yeniListem[j][k] == "kampanyalar":
                    yeniListem[j][k]=yeniListem[j][k].replace("kampanyalar", "kampanya")
        
        elif yeniListem[j][k] == "kampanyayı":
                    yeniListem[j][k]=yeniListem[j][k].replace("kampanyayı", "kampanya")
        
        elif yeniListem[j][k] == "teklifi":
                    yeniListem[j][k]=yeniListem[j][k].replace("teklifi", "teklif")
        
        elif yeniListem[j][k] == "indirimlerden":
                    yeniListem[j][k]=yeniListem[j][k].replace("indirimlerden", "indirim")
        
        elif yeniListem[j][k] == "kampanyalara":
                    yeniListem[j][k]=yeniListem[j][k].replace("kampanyalara", "kampanya")
        
        elif yeniListem[j][k] == "kuponları":
                    yeniListem[j][k]=yeniListem[j][k].replace("kuponları", "kupon")
        
        elif yeniListem[j][k] == "indirimlerimizi":
                    yeniListem[j][k]=yeniListem[j][k].replace("indirimlerimizi", "indirim")
        
        elif yeniListem[j][k] == "indirimlere":
                    yeniListem[j][k]=yeniListem[j][k].replace("indirimlere", "indirim")
        
        elif yeniListem[j][k] == "avantajlara":
                    yeniListem[j][k]=yeniListem[j][k].replace("avantajlara", "avantaj")
        
        elif yeniListem[j][k] == "kaçırmayın":
                    yeniListem[j][k]=yeniListem[j][k].replace("kaçırmayın", "kaçırma")
        
        elif yeniListem[j][k] == "kredinizi":
                    yeniListem[j][k]=yeniListem[j][k].replace("kredinizi", "kredi")
        
        elif yeniListem[j][k] == "hizmetleri":
                    yeniListem[j][k]=yeniListem[j][k].replace("hizmetleri", "hizmet")
        
        elif yeniListem[j][k] == "bonusunu":
                    yeniListem[j][k]=yeniListem[j][k].replace("bonusunu", "bonus")
        
        elif yeniListem[j][k] == "indiriminde":
                    yeniListem[j][k]=yeniListem[j][k].replace("indiriminde", "indirim")
        
        elif yeniListem[j][k] == "mağazalarımızda":
                    yeniListem[j][k]=yeniListem[j][k].replace("mağazalarımızda", "mağaza")