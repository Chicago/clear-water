$(function () {
    $('#chart').highcharts({
        chart: {
            type: 'column',
            height: 600,
            backgroundColor: '#242C33',
            style: {
                fontFamily: 'Poppins, sans-serif',
            }
        },
        colors: ['#21B2A6'],
        title: {
            text: 'Beach days when public<br /> is advised of poor water quality',
            style: {
                    color: "#fff"
            }
        },
        xAxis: {
            type: 'category',
            labels: {
                style: {
                    fontSize: '13px',
                    color: '#fff'
                }
            },
            title: {
                style: {
                    color: '#fff'
                }
            },
        },
        yAxis: {
            min: 0,
            title: {
                text: 'Beach Days',
                lineColor: "#fff",
                style: {
                    color: '#fff'
                },
                plotLines: [{
                    color: '#fff'
                }]
            },
            labels: {
                style: {
                    color: '#fff'
                }
            }
        },
        legend: {
            enabled: false
        },
        tooltip: {
            pointFormat: 'Number of beach days <br />an advisory was issued: <b>{point.y:f}</b>'
        },
        plotOptions: {
            column: {
                borderColor: 'none'
            }
        },
        exporting: {
            buttons: {
                contextButton: {
                    symbolStroke: '#242C33'
                }
            }
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
                    textShadow: '0 0 3px black'
                }
            }
        }]
    });
});