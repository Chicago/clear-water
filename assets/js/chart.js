$(function () {
    $('#chart').highcharts({
        chart: {
            type: 'column'
        },
        title: {
            text: 'Beach days when public<br /> is advised of poor water quality',
            style: {fontFamily: 'Rockwell, Rokkitt, Courier Bold, Courier, Georgia, Times, Times New Roman, serif',
                    fontWeight: 'bold'}
        },
        xAxis: {
            type: 'category',
            labels: {
                style: {
                    fontSize: '13px',
                    fontFamily: 'Futura, Futura-Medium, Futura Medium, Century Gothic, CenturyGothic, Apple Gothic, AppleGothic, URW Gothic L, Avant Garde, Questrial, sans-serif'
                }
            }
        },
        yAxis: {
            min: 0,
            title: {
                text: 'Beach Days',
                style: {fontFamily: 'Futura, Futura-Medium, Futura Medium, Century Gothic, CenturyGothic, Apple Gothic, AppleGothic, URW Gothic L, Avant Garde, Questrial, sans-serif'}
            }
        },
        legend: {
            enabled: false
        },
        tooltip: {
            pointFormat: 'Number of beach days <br />an advisory was issued: <b>{point.y:f}</b>'
        },
        series: [{
            name: 'Number of beach days',
            data: [
                ['Prior Approach', 9],
                ['New Approach', 69]
            ],
            dataLabels: {
                enabled: false,
                rotation: -90,
                color: '#FFFFFF',
                align: 'right',
                x: 4,
                y: 10,
                style: {
                    fontSize: '13px',
                    fontFamily: 'Futura, Futura-Medium, Futura Medium, Century Gothic, CenturyGothic, Apple Gothic, AppleGothic, URW Gothic L, Avant Garde, Questrial, sans-serif',
                    textShadow: '0 0 3px black'
                }
            }
        }]
    });
});