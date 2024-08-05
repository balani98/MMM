import { Component, OnInit } from '@angular/core';
import {
  faDashboard,
  faLocation,
  faShop,
  faBox,
  faMoneyBill,
  faChartBar,
  faContactBook,
  faHand,
  faMicrochip,
  faStethoscope
} from '@fortawesome/free-solid-svg-icons';
import { AnyObject } from 'chart.js/dist/types/basic';
import { ExplorerService } from 'src/services/explorer.service';

@Component({
  selector: 'app-side-nav',
  templateUrl: './side-nav.component.html',
  styleUrls: ['./side-nav.component.scss']
})
export class SideNavComponent implements OnInit {

  faDashboard = faDashboard;
  faLocation = faLocation;
  faShop = faShop;
  faBox = faBox;
  faMoneyBill = faMoneyBill;
  faChartBar = faChartBar;
  faContactBook = faContactBook;
  faHand = faHand;
  faMicrochip = faMicrochip;
  faStethoscope = faStethoscope
  selected:any = 'Insights'
  constructor(private explorerService:ExplorerService) { }
  select(item:any) {
    localStorage.setItem('selected', item);
    console.log(this)
    this.selected = item; 
  };
  isActive(item:any) {
    return this.selected === item;
  };
  ngOnInit(): void {
    this.selected = localStorage.getItem('selected')  
    this.select(this.selected);
  }

  downloadUserGuide() {
    this.explorerService.downloadUserguide().subscribe(
      (data) => {
        // Create a Blob object from the binary data
        const blob = new Blob([data], { type: 'application/pdf' });

        // Create a temporary URL for the Blob and trigger a download
        const downloadLink = document.createElement('a');
        downloadLink.href = window.URL.createObjectURL(blob);
        downloadLink.download = "userguide.pdf";
        downloadLink.click();
      },
      (error) => {
        console.error('Error downloading the PDF', error);
      }
    );
  }

}