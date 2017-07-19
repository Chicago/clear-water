/**
 * Parallax Scrolling Tutorial
 * For NetTuts+
 *  
 * Author: Mohiuddin Parekh
 *	http://www.mohi.me
 * 	@mohiuddinparekh   
 */

$(document).ready(function(){
    var isSmall = false;
 
   if (window.innerWidth >= 480){
       // Cache the Window object
        $window = $(window);
        $('section[data-type="background"]').each(function(){
            var $bgobj = $(this); // assigning the object
            $(window).scroll(function() {
                if(!isSmall){
                // Scroll the background at var speed
        	        // the yPos is a negative value because we're scrolling it UP!								
    		        var yPos = -($window.scrollTop() / $bgobj.data('speed')); 
    		
    		        // Put together our final background position
    		        var coords = '50% '+ yPos + 'px';
 
    		        // Move the background
    		        $bgobj.css({ backgroundPosition: coords });
                }
            }); // window scroll Ends
        });	
   }
 
    $(window).resize(function() {
        if(window.innerWidth <= 480)
            isSmall = true;
        else
            isSmall = false;
    });
 
}); 
 
 
 
/* 
 * Create HTML5 elements for IE's sake
 */
 
document.createElement("article");
document.createElement("section");
