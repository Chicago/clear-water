#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(ggplot2)

beaches = c("12th",
            "31st",
            "39th",
            "57th",
            "63rd",
            "Albion",
            "Calumet",
            "Jarvis",
            "Foster",
            "Osterman",
            "Howard",
            "Juneway",
            "Leone",
            "Montrose",
            "North Avenue",
            "Oak Street",
            "Ohio",
            "Rainbow",
            "Rogers",
            "South Shore"
)

# Define UI for application that draws a histogram
ui <- fluidPage(
  titlePanel("Can you predict better?"),
  
  sidebarLayout(
    
    sidebarPanel(tags$h4("Select which beaches will be predictive:"),
                 uiOutput("Box1"),
                 uiOutput("Box2"),
                 uiOutput("Box3"),
                 uiOutput("Box4"),
                 uiOutput("Box5"),
                 uiOutput("Box6"),
                 actionButton(inputId = "go", label = "Update")
                 
    ),
    
    mainPanel(
      verbatimTextOutput("default"),
      fluidRow(tags$h4("This can be a bar graph."), plotOutput("graph1")),
      fluidRow(tags$h4("This can be another bar graph."), plotOutput("graph2"))
    )
  )
  
)



# Define server logic required to draw a histogram
server <- function(input, output,session) {
  output$Box1 = renderUI(selectInput("choice1","Cluster 1",c(beaches), multiple = FALSE, selectize=FALSE))
  output$Box2 = renderUI(selectInput("choice2", "Cluster 2", c(setdiff(beaches, input$choice1)), multiple = FALSE, selectize=FALSE))
  output$Box3 = renderUI(selectInput("choice3", "Cluster 3", c(Reduce(setdiff,list(beaches, input$choice1, input$choice2))), multiple = FALSE, selectize=FALSE))
  output$Box4 = renderUI(selectInput("choice4", "Cluster 4", c(Reduce(setdiff,list(beaches, input$choice1, input$choice2, input$choice3))), multiple = FALSE, selectize=FALSE))
  output$Box5 = renderUI(selectInput("choice5", "Cluster 5", c(Reduce(setdiff,list(beaches, input$choice1, input$choice2, input$choice3, input$choice4))), multiple = FALSE, selectize=FALSE))
  output$Box6 = renderUI(selectInput("choice6", "Cluster 6", c(Reduce(setdiff,list(beaches, input$choice1, input$choice2, input$choice3, input$choice4, input$choice5))), multiple = FALSE, selectize=FALSE))
  data <- eventReactive(input$go, {functioncall(input$choice1,input$choice2, input$choice3, input$choice4, input$choice5, input$choice6)})
  
  output$default <- renderText({data()})
  
  output$graph1 <- renderPlot({bargraphstuff(algorithmdata)})
  output$graph2 <- renderPlot({bargraphstuff(algorithmdata)})
}

# Run the application 
shinyApp(ui = ui, server = server)

