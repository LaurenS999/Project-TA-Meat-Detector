package com.example.meatdetector_ta.viewmodel

import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.example.meatdetector_ta.R
import com.example.meatdetector_ta.model.informasi

class InformasiModel:ViewModel() {
    val informasiLD = MutableLiveData<List<informasi>>()

    fun refresh(){
        val informasi1 = informasi("Ayam",
                "1. Warna daging umumnya putih agak kemerahan.\n" +
                        "2. Serat daging halus\n" +
                        "3. Konsistensi kurang padat.\n" +
                        "4. Diantara serat daging tidak terdapat lemak.\n" +
                        "5. Warna lemak kekuning-kuningan dengan konsistensi lunak\n" +
                        "6. Bau agak amis sampai tidak berbau.",
                R.drawable.ayamsegar,
                R.drawable.ayamtidaksegar)

        val informasi2 = informasi("Babi",
                "1. Daging berwarna pucat hingga merah muda\n" +
                "2. Otot punggung yang banyak mengandung lemak, biasanya nampak kelabu putih\n" +
                "3. Daging berserat halus, konsistensi padat dan baunya spesifik\n" +
                "4. Pada umur tua, daging babi berwarna lebih tua, sedikit lemak dan serabut kasar\n" +
                "5. Lemak jauh lebih lembek dibanding lemak sapi atau kambing\n",
                R.drawable.babisegar,
                R.drawable.babitidaksegar)

        val informasi3 = informasi("Bebek",
                "1. Warna daging agak kemerahan..\n" +
                        "2. Daging bebek lebih liat dan basah, tetapi rasanya gurih dibandingkan dengan daging unggas lain.\n" +
                        "3. Daging bebek memiliki aroma yang lebih amis, sehingga penangannya tentu lebih rumit dibandingkan daging ayam " +
                        "Terutama untuk mengempukkan dagingnya yang liat serta menghilangkan aroma amis yang menyengat, dibutuhkan waktu dan pengalaman memasak.\n",
                R.drawable.bebeksegar,
                R.drawable.bebektidaksegar)

        val informasi4 = informasi("Kambing",
                "1. daging kambing adalah memiliki warna merah muda\n" +
                        "2. Daging berwarna lebih pucat dari daging domba\n" +
                        "3. Daging kambing jantan berbau khas\n" +
                        "4. serat yang lembut dan halus.\n" +
                        "5. Lemak daging kambing keras dan kenyal serta berwarna putih kekuningan.\n" +
                        "6. Di samping itu, aroma daging kambing lebih keras dibandingkan daging sapi.",
                R.drawable.kambingsegar,
                R.drawable.kambingtidaksegar)

        val informasi5 = informasi("Sapi",
                "1. Warna merah pucat, merah keungu-unguan atau kecoklatan dan akan berubah menjadi warna chery (merah cerah) bila daging tersebut kena oksigen.\n" +
                "2. Serabut daging halus tapi tidak mudah hancur dan sedikit berlemak\n" +
                "3. Konsistensi liat, jika saat dicubit seratnya terlepas maka daging sudah tidak baik.\n" +
                "4. Lemak berwarna kekuning-kuningan\n" +
                "5. Tekstur daging kenyal.\n" +
                "6. Kondisi daging sapi keras namun tidak kaku.\n" +
                "7. Bau dan rasa aromatis.\n",
                R.drawable.sapisegar,
                R.drawable.sapitidaksegar)

        val informasiList:ArrayList<informasi> = arrayListOf<informasi>(informasi1, informasi2, informasi3, informasi4, informasi5)

        informasiLD.value = informasiList
    }
}