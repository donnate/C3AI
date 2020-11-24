# Define UI for app that draws a histogram ----
library("shiny")

setwd("/Users/cdonnat/Dropbox/COVID-app2/")
VIRTUALENV_NAME = 'python_environment'
countries = read.csv("population_by_country_2020.csv")
regions = read.csv("Regions.csv")
countries_list = list()
for (c in countries$Country){
  if (c %in% regions$country ){
    countries_list[[c]]= as.list(sapply(regions$region[regions$country == c], function(x){toString(x)}))
  }else{
    countries_list[[c]]= list(c(""))
  }
}
COVID_report = read.csv("Report_data.csv", header=T)
PYTHON_DEPENDENCIES = c('numpy', 'setuptools==49.3.0', 'scipy', 'pandas==0.23.3', 'requests', 'matplotlib')
# ------------------------- Settings (Do not edit) -------------------------- #

if (Sys.info()[['user']] == 'shiny'){
  # Running on shinyapps.io
  Sys.setenv(PYTHON_PATH = 'python3')
  Sys.setenv(VIRTUALENV_NAME = VIRTUALENV_NAME)# Installs into default shiny virtualenvs dir
  Sys.setenv(RETICULATE_PYTHON = paste0('/home/shiny/.virtualenvs/', VIRTUALENV_NAME, '/bin/python'))
} else if (Sys.info()[['user']] == 'rstudio-connect'){
  
  # Running on remote server
  Sys.setenv(PYTHON_PATH = '/opt/python/3.7.6/bin/python')
  Sys.setenv(VIRTUALENV_NAME = paste0(VIRTUALENV_NAME, '/')) # include '/' => installs into rstudio-connect/apps/
  Sys.setenv(RETICULATE_PYTHON = paste0(VIRTUALENV_NAME, '/bin/python'))
  
} else {
  
  # Running locally
  options(shiny.port = 7450)
  Sys.setenv(PYTHON_PATH = 'python3')
  Sys.setenv(VIRTUALENV_NAME = VIRTUALENV_NAME) # exclude '/' => installs into ~/.virtualenvs/
}

virtualenv_dir = Sys.getenv('VIRTUALENV_NAME')
print(virtualenv_dir)
python_path = Sys.getenv('PYTHON_PATH')

reticulate::virtualenv_create(envname = virtualenv_dir, python = python_path)
reticulate::virtualenv_install(virtualenv_dir, packages = PYTHON_DEPENDENCIES)
reticulate::use_virtualenv(virtualenv_dir, required = T)
print(reticulate::py_discover_config())


PYTHON_DEPENDENCIES = c('numpy', 'pandas', 'scipy','scikit-learn')
#condaenv_create(envname = "python_environment", python= "python3")
#condaenv_install("python_environment", packages = c('pandas','numpy','scipy','scikit-learn'))
#use_condaenv("python_environment", required = TRUE)
#condaenv_create(envname = "/Users/cdonnat/Dropbox/COVID-app/python_environment", python= "python3")
# Explicitly install python libraries that you want to use, e.g. pandas, numpy
#condaenv_install("/Users/cdonnat/Dropbox/COVID-app/python_environment", packages = c('pandas','numpy'))
# Select the conda environmentSys.info()[['user']]




ui <- fluidPage(
  
  # App title ----
  titlePanel("What is the probability that I've had COVID"),
  
  actionButton("button", "An action button"),
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    
    # Sidebar panel for inputs ----
    sidebarPanel(
      
      # Input: Slider for the number of bins ----
      #radioButtons(inputId = "test",
      #            label = "Have you taken an antibody test before?",
      #             choices =  c( "No"= 0,"Yes" = 1), inline = TRUE),
      radioButtons(inputId = "test",
                   label = "If you have taken an antibody test, what was your result?",
                   choices =  c(  "Negative (only the control line)"= 0,
                                  "Positive (Control + IgG and/or IgM line)" = 1,
                                  "I haven't taken any antibody tests"= 2),
                   selected = 2,
                   inline = FALSE),
      dateInput(inputId = "date_symptoms",
                label ="If applicable, when did the symptoms start?", 
                value ="2020-09-01",
                min = NULL,
                max = NULL,
                format = "yyyy-mm-dd",
                startview = "month",
                weekstart = 0,
                language = "en",
                width = NULL,
                autoclose = TRUE,
                datesdisabled = NULL,
                daysofweekdisabled = NULL
      ),
      radioButtons(inputId = "label_t",
                   label = "If applicable, how long after the end of your symptoms did you take the test?",
                   choices =  c( "NA"= 'missing', 
                                 "I've never felt ill"='asymptomatic', 
                                 "Less than 10 days"= "2-10 days",
                                 '11-20 days' = '11-20 days',
                                 '21+ days' = '21+ days'), inline = FALSE),
      radioButtons(inputId = "chills",
                   label = "Did you experience any chills?",
                   choices =  c( "No"= 0,"Yes" = 1), inline = TRUE),
      radioButtons(inputId = "shortnessOfBreath",
                   label = "Shortness Of Breath?",
                   choices =  c( "No"= 0,"Yes" = 1),inline = TRUE),
      radioButtons(inputId = "achyJointsMuscles",
                   label = "Achy Joints or Muscles?",
                   choices =  c( "No"= 0,"Yes" = 1), inline = TRUE),
      radioButtons(inputId = "fever",
                   label = "A fever/ high temperature?",
                   choices =c( "No"= 0,"Yes" = 1), inline = TRUE),
      radioButtons(inputId = "headache",
                   label = "Headaches?",
                   choices = c( "No"= 0,"Yes" = 1), inline = TRUE),
      radioButtons(inputId = "congestedNose",
                   label = "A congested nose?",
                   choices =c( "No"= 0,"Yes" = 1), inline = TRUE),
      radioButtons(inputId = "soreThroat",
                   label = "A sore throat?",
                   choices = c( "No"= 0,"Yes" = 1), inline = TRUE),
      radioButtons(inputId = "lossTasteSmell",
                   label = "Any loss of taste or smell?",
                   choices =  c( "No"= 0,"Yes" = 1), inline = TRUE),
      radioButtons(inputId = "dryCough",
                   label = "A dry cough?",
                   choices = c( "No"= 0,"Yes" = 1), inline = TRUE),
      radioButtons(inputId = "coughWithSputum",
                   label = "A cough with sputum?",
                   choices =  c( "No"= 0,"Yes" = 1),  inline = TRUE),
      radioButtons(inputId = "fatigue",
                   label = "Fatigue?",
                   choices =  c( "No"= 0,"Yes" = 1), inline = TRUE),
      radioButtons(inputId = "nausea",
                   label = "Nausea?",
                   choices =  c( "No"= 0,"Yes" = 1), inline = TRUE),
      radioButtons(inputId = "stomachUpsetDiarrhoea",
                   label = " An upset stomach/ dirrhoea?",
                   choices =  c( "No"= 0,"Yes" = 1), inline = TRUE),
      radioButtons(inputId = "confusion",
                   label = "Confusion?",
                   choices = c( "No"= 0,"Yes" = 1),inline = TRUE),
      sliderInput(inputId = "howAnxious",
                  label = "How anxious were you?",
                  min = 0,
                  max = 10,
                  value = 5),
      sliderInput(inputId = "howBadDidTheyFeel",
                  label = "How bad did/do you feel?",
                  min = 0,
                  max = 10,
                  value = 5),
      sliderInput(inputId = "howShortOfBreath",
                  label = "How short of breath did/do you feel?",
                  min = 0,
                  max = 10,
                  value = 5),
      selectInput(inputId = "index_lengthOfTimeOfSymptoms",
                  label = "How long did the symptoms last?",
                  choices = c('I never felt ill' = 0, "Less than a week" = 1, "One to two weeks"=2, "More than two weeks"=3,
                              'More than three weeks' = 4),selected = 2),
      sliderInput(inputId = "numberInHousehold",
                  label = "How many other people do you live with?",
                  min = 0,
                  max = 20,
                  value = 2),
      radioButtons(inputId = "householdIllness",
                   label = "Did anyone else in your household fall ill",
                   choices = c("No"= 0, "Yes" = 1), inline = TRUE),
      sliderInput(inputId = "numberIllInHousehold",
                  label = "If applicable, how many people in your household fell ill?",
                  min = 0,
                  max = 20,
                  value = 0),
      selectInput(inputId = "country",
                  label = "Where do you live?",
                  choices = countries$Country,
                  selected = "United Kingdom"),
      selectInput(inputId = "region",
                  label = "Where do you live?",
                  choices = countries_list),
      uiOutput("region")
      #submitButton("Calculate", icon("refresh"))
    ),
    
    
    
    # Main panel for displaying outputs ----
    mainPanel(
      
      # Output: Histogram ----
      tabsetPanel(
        tabPanel("Disclaimer", htmlOutput("disclaimer")),
        tabPanel("Plot", plotOutput("distPlot")),
        #tabPanel("Table", tableOutput("probs")),
        tabPanel("Report", h3(htmlOutput("Report"))))
      
    )
  )
)


print(Sys.which("python"))
reticulate::source_python('python_script.py')
# Define server logic required to draw a histogram ----
server <- function(input, output, session) {
  
  # Histogram of the Old Faithful Geyser Data ----
  # with requested number of bins
  # This expression that generates a histogram is wrapped in a call
  # to renderPlot to indicate that:
  #
  # 1. It is "reactive" and therefore should be automatically
  #    re-executed when inputs (input$bins) change
  # 2. Its output type is a plot
  # ------------------ App virtualenv setup (Do not edit) ------------------- #

  
  dataInput <- observeEvent(input$do, {
    testMethod(achyJointsMuscles= as.numeric(input$achyJointsMuscles),
               chills = as.numeric(input$chills),
               confusion = as.numeric(input$confusion),
               congestedNose = as.numeric(input$congestedNose),
               coughWithSputum= as.numeric(input$coughWithSputum),
               dryCough = as.numeric(input$dryCough),
               fatigue = as.numeric(input$fatigue),
               fever = as.numeric(input$fever),
               headache = as.numeric(input$headache), 
               lossTasteSmell = as.numeric(input$lossTasteSmell),
               nausea = as.numeric(input$nausea),
               shortnessOfBreath = as.numeric(input$shortnessOfBreath), 
               soreThroat = as.numeric(input$soreThroat), 
               stomachUpsetDiarrhoea = as.numeric(input$stomachUpsetDiarrhoea),
               householdIllness = as.numeric(input$householdIllness),
               numberInHousehold = as.numeric(input$numberInHousehold), 
               numberIllInHousehold = as.numeric(input$numberIllInHousehold), 
               percentage_householdIllness = as.numeric(input$numberIllInHousehold)/(1+as.numeric(input$numberInHousehold)),
               howAnxious = as.numeric(input$howAnxious), 
               howBadDidTheyFeel=  as.numeric(input$howBadDidTheyFeel),
               howShortOfBreath = as.numeric(input$howShortOfBreath),
               index_lengthOfTimeOfSymptoms = as.numeric(input$index_lengthOfTimeOfSymptoms),
               date_symptoms = input$date_symptoms,
               label_t = input$label_t,
               test = as.numeric(input$test),
               symptomatic =  as.numeric(input$achyJointsMuscles) +  as.numeric(input$chills) + as.numeric(input$confusion) + as.numeric(input$congestedNose) +
                 as.numeric(input$coughWithSputum)+ as.numeric(input$dryCough)+ as.numeric(input$fatigue) + as.numeric(input$fever) + as.numeric(input$headache) +
                 as.numeric(input$lossTasteSmell)+ as.numeric(input$nausea) +  as.numeric(input$shortnessOfBreath) + as.numeric(input$soreThroat) + as.numeric(input$stomachUpsetDiarrhoea),
               country= input$country,
               region = input$region
    )
  })
  
  
  
  output$distPlot <- renderPlot({
    
    x =  dataInput()
    #### Write the report
    q = quantile(x, probs =c(0.1, 0.5, 0.90))
    q[2] = mean(x)
    conclusion = ""
    if (q[2]> 0.5){
      conclusion = paste0("According to your questionnaire,  you are more likely to have COVID-19 specific antibodies (probability= ", round(q[2],2),  "). \n \n")
      if(q[1] < 0.5){
        conclusion = paste0( conclusion, "The uncertainty associated with this result is high (CI: [" , round(q[1],2), "," , round(q[3],2), "])." )
      }else{
        conclusion = paste0(conclusion, "The evidence provided by your answers is significant, and the algorithm is fairly confident (CI: [" , round(q[1],2), "," , round(q[3],2), "])." )
      }
    }else{
      conclusion = paste0("According to your questionnaire,  you are more likely to NOT have COVID-19 specific antibodies (probability = ", round(q[2],2),  "). \n \n")
      if (q[3] > 0.5){
        conclusion = paste0(conclusion, "The uncertainty associated with this result is high (CI: [" , round(q[1],2), "," , round(q[3],2), "])." )
      }else{
        conclusion = paste0(conclusion, "The evidence provided by your answers is significant, and the algorithm is fairly confident (CI: [" , round(q[1],2), "," , round(q[3],2), "])." )
      }
    }
    hist(x, breaks = seq(from=0, to=1, by=0.025), col = "#75AADB", border = "white",
         xlab = "Probability",
         main = conclusion)
    
  })
  

  
}


shinyApp(ui, server)
