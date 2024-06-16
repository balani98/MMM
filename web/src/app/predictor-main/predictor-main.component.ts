import { HttpResponse } from '@angular/common/http';
import { Component, setTestabilityGetter } from '@angular/core';
import * as d3 from 'd3';
import { PredictorService } from 'src/services/predictor.service';
@Component({
  selector: 'app-predictor-main',
  templateUrl: './predictor-main.component.html',
  styleUrls: ['./predictor-main.component.scss'],
})
export class PredictorMainComponent {
  constructor(private predictorService: PredictorService) {} 
  predictorInputs:boolean=false;
  tableData: any = {};
  tableDataKeys:any = []
  setTableData(effectiveSharesData:any ){
    this.tableData = effectiveSharesData;
    this.tableDataKeys = Object.keys(this.tableData)
    this.predictorInputs=true;
  }
  
  getReponseCurves(){
    this.predictorService.generateRespCurves().subscribe({
        next: async(res: any) => {
          if(res.status == 200){
            this.predictorInputs = true
            var responseCurvesJson = await res.body.output_dict.response_curve
            var effectiveSharesJson = await res.body.output_dict.effective_share
            //console.log('deeps',effectiveSharesJson)
            var responseCurveStats = await res.body.output_dict.response_curve_stats
            this.plot_multi_line_chart(responseCurvesJson, responseCurveStats)
            this.setTableData(effectiveSharesJson)
          }
        }, error: (errorResponse) => {

          if(errorResponse.status == 500)
            alert("No model exists for showing results")
          else if(errorResponse.status == 400)
            alert("Model is in training phase")
        }
      });
  }
  // plot multiple line charts with checkboxes
  plot_multi_line_chart(plotting_data: any,responseCurveStats:any) {
    console.log(plotting_data);
    var x_label = 'Investment';
    var margin = { top: 10, right: 50, bottom: 50, left: 70 },
      currentWidth = parseInt(
        d3.select('#multi_line_chart_with_checkboxes').style('width')
      );
    var width = currentWidth - margin.left - margin.right;
    var height = 500 - margin.top - margin.bottom;
    //initialize scales
    // If svg exists before remove
    d3.select('#multi_line_chart_with_checkboxes').select('svg').remove();
    var svg = d3
      .select('#multi_line_chart_with_checkboxes')
      .append('svg')
      .attr(
        'viewBox',
        `0 0 ${width + margin.left + margin.right} ${
          height + margin.top + margin.bottom
        }`
      )
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
      .style('fill', '#ced4da');
    const dimension_groups = plotting_data.map((obj: any) => {
      return obj.name;
    });
    var myColor = d3.scaleOrdinal(d3.schemeDark2).domain(dimension_groups);
    //.range(d3.schemeSet1)
    var x = d3
      .scaleLinear()
      .domain([0, responseCurveStats.spend[1] + 0.1*responseCurveStats.spend[1]])
      .range([0, width]);
    // adding dollars to x axis
    svg
      .append('g')
      .attr('transform', 'translate(0,' + height + ')')
      .call(
        d3.axisBottom(x).tickFormat(function (d) {
          if (x_label != 'Impression')
            return (
              '＄' +
              d.toString().replace(/\B(?<!\.\d*)(?=(\d{3})+(?!\d))/g, ',')
            );
          else
            return d.toString().replace(/\B(?<!\.\d*)(?=(\d{3})+(?!\d))/g, ',');
        })
      );
    // text label for x axis
    svg
      .append('text')
      .attr(
        'transform',
        'translate(' + width / 2 + ' ,' + (height + margin.top + 30) + ')'
      )
      .style('text-anchor', 'middle')
      .style('fill', 'black')
      .text(x_label);

    // just for text label for the y axis
    var y_label = 'target';
    svg
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - height / 2)
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .style('fill', 'black')
      .text(y_label);

    var y = d3
      .scaleLinear()
      .domain([0, responseCurveStats.target[1] + 0.1 *responseCurveStats.target[1]])
      .range([height, 0]);
    // adding dollars to y axis
    svg.append('g').call(
      d3.axisLeft(y).tickFormat(function (d) {
        return d.toString().replace(/\B(?<!\.\d*)(?=(\d{3})+(?!\d))/g, ',');
      })
    );

    var line = d3
      .line()
      .x(function (d: any) {
        return x(d.spend);
      })
      .y(function (d: any) {
        return y(d.target);
      });
    svg
      .selectAll('myLines')
      .data(plotting_data)
      .join('path')
      .attr('id', function (d: any) {
        return (
          'line-' +
          d.name
            .replace(/\s/g, '')
            .replace(/[^\w\s]/gi, '')
            .replace('/', '')
        );
      })
      .attr('d', function (d: any) {
        return line(d.values);
      })
      .attr('stroke', function (d: any) {
        return myColor(d.name);
      })
      .style('stroke-width', 3)
      .style('fill', 'none')
      .style('sharp-rendering', 'auto');
    // Add the label at the end of each line
    svg
      .selectAll('myLabels')
      .data(plotting_data)
      .enter()
      .append('g')
      .append('text')
      .attr('class', function (d: any) {
        return d.name;
      })
      .attr('id', function (d: any) {
        return (
          'text-' +
          d.name
            .replace(/\s/g, '')
            .replace(/[^\w\s]/gi, '')
            .replace('/', '')
        );
      })
      .datum(function (d: any) {
        return {
          name: d.name,
          value: d.values[d.values.length - 1], // keep only the last values of each time series
        };
      })
      .attr('transform', function (d: any) {
        console.log(d.value);
        return 'translate(' + x(d.value.spend) + ',' + y(d.value.target) + ')';
      })
      .attr('x', 12) // shift the text a bit more right
      .text(function (d) {
        return d.name;
      })
      .style('fill', function (d) {
        return myColor(d.name);
      })
      .style('font-size', 15);
    var x_range = d3.scaleLinear().range([0, width]);
    var y_range = d3.scaleLinear().range([height, 0]);
    for (let i = 0; i < height; i += 80) {
      svg
        .append('line')
        .attr('x1', 0)
        .attr('y1', i)
        .attr('x2', width)
        .attr('y2', i)
        .style('stroke', 'lightgray');
    }
    // Draw vertical lines
    for (let i = 0; i < width; i += 80) {
      svg
        .append('line')
        .attr('x1', i)
        .attr('y1', 0)
        .attr('x2', i)
        .attr('y2', height)
        .style('stroke', 'lightgray');
    }
  }
  ngOnInit(): void {
    console.log('ngOnInit called'); // Check if this is logged
    this.getReponseCurves()
  }
}
