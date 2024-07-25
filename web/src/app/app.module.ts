import { NgModule, forwardRef } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HeaderComponent } from './header/header.component';
import { SideNavComponent } from './side-nav/side-nav.component';
import { MainComponent } from './main/main.component';
import { TopWidgetsComponent } from './top-widgets/top-widgets.component';
import { SalesByMonthComponent } from './sales-by-month/sales-by-month.component';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { ExplorerFormComponent } from './explorer-form/explorer-form.component';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { MatTabsModule } from '@angular/material/tabs';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import { ProfilingStatsComponent } from './profiling-stats/profiling-stats.component';
import { ChannelProfilingStatsComponent } from './channel-profiling-stats/channel-profiling-stats.component';
import { ChannelQuantileStatsComponent } from './channel-quantile-stats/channel-quantile-stats.component';
import { NavbarComponent } from './navbar/navbar.component';
import { PredictorMainComponent } from './predictor-main/predictor-main.component';
import { PredictorFormComponent } from './predictor-form/predictor-form.component';
// import { ChartModule } from 'angular-highcharts';
import {MatSelectModule} from '@angular/material/select';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatCheckboxModule} from '@angular/material/checkbox';
import { ShortNumberPipe } from 'src/pipes/shortnumber.pipe';
import { ConvertToPercentagePipe } from 'src/pipes/converttopercentage.pipe';
import { DecimalPipe } from '@angular/common';
import { NgxSpinnerModule } from 'ngx-spinner';
import { CurrencySymbolPipe } from 'src/pipes/currency-symbol.pipe';
import { RoundingPipe } from 'src/pipes/rounding.pipe';
import { ValidationReportStatsComponent } from './validation-report-stats/validation-report-stats.component';
import { ModelHealthComponent } from './model-health/model-health.component';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    SideNavComponent,
    MainComponent,
    TopWidgetsComponent,
    SalesByMonthComponent,
    ExplorerFormComponent,
    ProfilingStatsComponent,
    ChannelProfilingStatsComponent,
    ChannelQuantileStatsComponent,
    NavbarComponent,
    PredictorMainComponent,
    PredictorFormComponent,
    ShortNumberPipe,
    ConvertToPercentagePipe,
    CurrencySymbolPipe,
    RoundingPipe,
    ValidationReportStatsComponent,
    ModelHealthComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FontAwesomeModule,
    HttpClientModule,
    // ChartModule
    FormsModule,
    ReactiveFormsModule,
    MatTabsModule,
    BrowserAnimationsModule,
    MatSelectModule,
    MatFormFieldModule,
    MatCheckboxModule,
    NgxSpinnerModule.forRoot({ type: 'ball-scale-multiple' })
  ],
  providers: [DecimalPipe,RoundingPipe, provideAnimationsAsync()],
  bootstrap: [AppComponent]
})
export class AppModule { }
