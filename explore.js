/*global angular,Plotly*/

angular.module('explore', []);

angular.module('explore').controller('explore', ['$scope', '$http', function ($scope, $http) {
    $scope.error = '';

    $scope.get = function (url, success, failure) {
        $http({ url: url }).then(
            function (res) {
                if (success) {
                    success(res.data);
                }
            },
            
            function (trace) {        
                $scope.error = trace;
                if (failure) {
                    failure();
                }
            }
        );
    };
}]);

angular.module('explore').directive('chart', [function () {
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