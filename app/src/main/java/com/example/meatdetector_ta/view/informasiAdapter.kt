package com.example.meatdetector_ta.view

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.meatdetector_ta.R
import com.example.meatdetector_ta.model.informasi
import kotlinx.android.synthetic.main.informasi_item.view.*
import java.io.InputStream

class informasiAdapter(val informationlist:ArrayList<informasi>)
    :RecyclerView.Adapter<informasiAdapter.InformasiViewHolder>()
{
    class InformasiViewHolder(var view: View) : RecyclerView.ViewHolder(view)

    fun refreshData(newInfomasiList: List<informasi>)
    {
        informationlist.clear()
        informationlist.addAll(newInfomasiList)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): InformasiViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        val view = inflater.inflate(R.layout.informasi_item, parent, false)
        return InformasiViewHolder(view)
    }

    override fun onBindViewHolder(holder: InformasiViewHolder, position: Int) {
        holder.view.txtjenisdaging.text = "Daging " + informationlist[position].jenisDaging
        holder.view.txtDeskripsiDaging.text = informationlist[position].deskripsi_daging

        holder.view.imageViewDagingSegar.setImageResource(informationlist[position].url_daging_segar)
        holder.view.imageViewDagingTidakSegar.setImageResource(informationlist[position].url_daging_tidaksegar)
    }

    override fun getItemCount(): Int {
        return informationlist.count()
    }

}