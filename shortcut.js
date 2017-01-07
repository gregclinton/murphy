// usage: <shortcut key="r" keypress="get('/optimize/reload')"></shortcut>

angular.module('main').directive('shortcut', ['$document', function ($document) {
    return {
        restrict: 'E',
        replace: true,
        scope: true,
        link: function (scope, element, attrs) {
            function code(key) {
                return key === 'enter' ? 13 : key === 'backspace' ? 8 : key.charCodeAt(0);
            }
            
            $document.bind('keypress', function (e) {
                if (code(attrs.key) === e.which) {
                    scope.$apply(attrs.keypress);
                }
            });
            
            $document.bind('keydown', function (e) {
                if (code(attrs.key) === e.which) {
                    scope.$apply(attrs.keydown);
                }
            });
        }
    };
}]);