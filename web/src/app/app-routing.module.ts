import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { NavbarComponent } from './navbar/navbar.component';
import { MainComponent } from './main/main.component';
import { PredictorMainComponent } from './predictor-main/predictor-main.component';
import { PredictorFormComponent } from './predictor-form/predictor-form.component';
import { ModelHealthComponent } from './model-health/model-health.component';

const routes: Routes = [
  {
    path: '',
    component: NavbarComponent,
    children: [
      {
        path: '',
        redirectTo: 'explorer',
        pathMatch: 'full',
      },
      {
        path: 'explorer',
        component: MainComponent,
      },
      {
        path: 'buildModel',
        component: PredictorFormComponent,
      },
      {
        path: 'showResults',
        component: PredictorMainComponent,
      },
      {
        path: 'modelHealth',
        component: ModelHealthComponent,
      },
    ],
  },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
