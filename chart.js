/*global angular,Plotly*/

angular.module('main').directive('chart', [function () {
    function link(scope, element, attribute) {
        var json = scope.data,
            options = { displayModeBar: false, staticPlot: true };
        
        Plotly.plot(element[0], json.data, json.layout, options);
    }    
    
    return {
        restrict: 'E',
        scope: { data: '=' },
        template: '<span style="outline: none; border: 0;"> </span>',
        link: link
    };
}]);